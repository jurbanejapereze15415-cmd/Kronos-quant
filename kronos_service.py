"""
Servicio de Inferencia de Kronos (Conexión directa con shiyu-coder/Kronos y Hugging Face)
Carga el modelo fundacional financiero Kronos en el Mac (acelerado con Apple Silicon MPS o CPU)
y genera múltiples trayectorias probabilísticas de velas futuras (K-lines) + Abanico Monte Carlo.
"""

import os
import sys
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch

# Añadir la carpeta del repositorio clonado Kronos al path de Python
KRONOS_REPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Kronos")
if KRONOS_REPO_PATH not in sys.path:
    sys.path.insert(0, KRONOS_REPO_PATH)

from model import Kronos, KronosTokenizer
from model.kronos import calc_time_stamps, sample_from_logits


class KronosEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id = "NeoQuasar/Kronos-mini"
        self.tokenizer_id = "NeoQuasar/Kronos-Tokenizer-2k"
        self.max_context = 512
        self.clip = 5.0
        if torch.cuda.is_available():
            self.device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.is_loaded = False

    def ensure_loaded(self, model_size: str = "mini"):
        target_model = "NeoQuasar/Kronos-small" if model_size == "small" else "NeoQuasar/Kronos-mini"
        target_tok = (
            "NeoQuasar/Kronos-Tokenizer-base"
            if model_size == "small"
            else "NeoQuasar/Kronos-Tokenizer-2k"
        )
        if self.is_loaded and self.model_id == target_model:
            return

        print(f"[KronosEngine] Cargando {target_model} y {target_tok} en dispositivo '{self.device}'...")
        self.tokenizer = KronosTokenizer.from_pretrained(target_tok)
        self.model = Kronos.from_pretrained(target_model)
        self.tokenizer.eval().to(self.device)
        self.model.eval().to(self.device)
        self.model_id = target_model
        self.tokenizer_id = target_tok
        self.is_loaded = True
        print(f"[KronosEngine] Modelo {target_model} listo en '{self.device}'.")

    def predict_multi_scenarios(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int = 16,
        sample_count: int = 24,
        temperature: float = 1.0,
        top_p: float = 0.92,
        model_size: str = "mini",
    ) -> Dict[str, Any]:
        """
        Ejecuta Kronos generando `sample_count` escenarios futuros en paralelo
        para obtener:
        - Abanico completo de trayectorias Monte Carlo (como en KRONOS FORECAST)
        - Las velas medianas predichas (Open, High, Low, Close, Volume)
        - Percentiles p90, med (p50) y p10
        - Régimen (BULLISH / BEARISH / CHOPPY según p_up)
        """
        self.ensure_loaded(model_size=model_size)

        price_cols = ["open", "high", "low", "close"]
        vol_col = "volume"
        amt_col = "amount"

        work_df = df.copy().tail(min(len(df), 380)).reset_index(drop=True)
        x_ts = x_timestamp.tail(len(work_df)).reset_index(drop=True)
        y_ts = y_timestamp.reset_index(drop=True)

        if vol_col not in work_df.columns:
            work_df[vol_col] = 0.0
        if amt_col not in work_df.columns:
            work_df[amt_col] = work_df[vol_col] * work_df[price_cols].mean(axis=1)

        x_time_df = calc_time_stamps(x_ts)
        y_time_df = calc_time_stamps(y_ts)

        x_raw = work_df[price_cols + [vol_col, amt_col]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)[np.newaxis, :]
        y_stamp = y_time_df.values.astype(np.float32)[np.newaxis, :]

        x_mean = np.mean(x_raw, axis=0)
        x_std = np.std(x_raw, axis=0)
        x_norm = (x_raw - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -self.clip, self.clip)[np.newaxis, :]

        with torch.no_grad():
            x_tensor = torch.from_numpy(x_norm).to(self.device)
            x_stamp_t = torch.from_numpy(x_stamp).to(self.device)
            y_stamp_t = torch.from_numpy(y_stamp).to(self.device)

            x_rep = x_tensor.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_tensor.size(1), x_tensor.size(2))
            x_stamp_rep = x_stamp_t.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp_t.size(1), x_stamp_t.size(2))
            y_stamp_rep = y_stamp_t.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp_t.size(1), y_stamp_t.size(2))

            x_token = self.tokenizer.encode(x_rep, half=True)
            initial_seq_len = x_rep.size(1)
            batch_size = x_token[0].size(0)
            total_seq_len = initial_seq_len + pred_len
            full_stamp = torch.cat([x_stamp_rep, y_stamp_rep], dim=1)

            generated_pre = x_token[0].new_empty(batch_size, pred_len)
            generated_post = x_token[1].new_empty(batch_size, pred_len)

            pre_buffer = x_token[0].new_zeros(batch_size, self.max_context)
            post_buffer = x_token[1].new_zeros(batch_size, self.max_context)
            buffer_len = min(initial_seq_len, self.max_context)
            if buffer_len > 0:
                start_idx = max(0, initial_seq_len - self.max_context)
                pre_buffer[:, :buffer_len] = x_token[0][:, start_idx : start_idx + buffer_len]
                post_buffer[:, :buffer_len] = x_token[1][:, start_idx : start_idx + buffer_len]

            for i in range(pred_len):
                current_seq_len = initial_seq_len + i
                window_len = min(current_seq_len, self.max_context)

                if current_seq_len <= self.max_context:
                    input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
                else:
                    input_tokens = [pre_buffer, post_buffer]

                context_end = current_seq_len
                context_start = max(0, context_end - self.max_context)
                current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

                s1_logits, context = self.model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
                s1_logits = s1_logits[:, -1, :]
                sample_pre = sample_from_logits(s1_logits, temperature=temperature, top_k=0, top_p=top_p, sample_logits=True)

                s2_logits = self.model.decode_s2(context, sample_pre)
                s2_logits = s2_logits[:, -1, :]
                sample_post = sample_from_logits(s2_logits, temperature=temperature, top_k=0, top_p=top_p, sample_logits=True)

                generated_pre[:, i] = sample_pre.squeeze(-1)
                generated_post[:, i] = sample_post.squeeze(-1)

                if current_seq_len < self.max_context:
                    pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                    post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
                else:
                    pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                    post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                    pre_buffer[:, -1] = sample_pre.squeeze(-1)
                    post_buffer[:, -1] = sample_post.squeeze(-1)

            full_pre = torch.cat([x_token[0], generated_pre], dim=1)
            full_post = torch.cat([x_token[1], generated_post], dim=1)

            context_start = max(0, total_seq_len - self.max_context)
            input_tokens = [
                full_pre[:, context_start:total_seq_len].contiguous(),
                full_post[:, context_start:total_seq_len].contiguous(),
            ]
            z = self.tokenizer.decode(input_tokens, half=True)
            z_pred = z[:, -pred_len:, :].cpu().numpy()

        # Desnormalizar todas las trayectorias: (sample_count, pred_len, 6)
        all_paths = z_pred * (x_std + 1e-5) + x_mean

        median_path = np.median(all_paths, axis=0)
        p90_band = np.percentile(all_paths[:, :, 3], 90, axis=0)  # Percentil 90 del Close
        p10_band = np.percentile(all_paths[:, :, 3], 10, axis=0)  # Percentil 10 del Close

        last_real_close = float(work_df["close"].iloc[-1])

        # Calcular cuántos escenarios terminan por encima del precio actual (p_up)
        final_closes = all_paths[:, -1, 3]
        paths_up = int(np.sum(final_closes > last_real_close))
        p_up = paths_up / float(sample_count)
        bull_prob_pct = round(p_up * 100.0, 1)

        # Régimen estilo run_kronos.py (0.35 < p_up < 0.65 -> CHOPPY)
        if 0.35 < p_up < 0.65:
            regime = "CHOPPY"
        elif p_up >= 0.65:
            regime = "BULLISH"
        else:
            regime = "BEARISH"

        pred_candles = []
        for idx in range(pred_len):
            o_val = float(median_path[idx, 0])
            h_val = float(median_path[idx, 1])
            l_val = float(median_path[idx, 2])
            c_val = float(median_path[idx, 3])
            v_val = max(0.0, float(median_path[idx, 4]))

            real_h = max(o_val, c_val, h_val)
            real_l = min(o_val, c_val, l_val)
            ts = int(pd.Timestamp(y_ts.iloc[idx]).timestamp())

            pred_candles.append({
                "time": ts,
                "open": round(o_val, 4),
                "high": round(real_h, 4),
                "low": round(real_l, 4),
                "close": round(c_val, 4),
                "volume": round(v_val, 2),
                "upper_band": round(float(p90_band[idx]), 4),
                "lower_band": round(float(p10_band[idx]), 4),
            })

        # Extraer las trayectorias individuales de cierre para dibujar el abanico Monte Carlo
        monte_carlo_paths = []
        for s_idx in range(sample_count):
            path_points = [round(float(all_paths[s_idx, t_idx, 3]), 4) for t_idx in range(pred_len)]
            monte_carlo_paths.append(path_points)

        pred_final_close = pred_candles[-1]["close"]
        expected_return_pct = round(((pred_final_close - last_real_close) / max(last_real_close, 1e-8)) * 100.0, 2)

        max_pred_high = max(c["high"] for c in pred_candles)
        min_pred_low = min(c["low"] for c in pred_candles)
        volatility_range_pct = round(((max_pred_high - min_pred_low) / max(last_real_close, 1e-8)) * 100.0, 2)

        max_p90_all = float(np.max(p90_band))
        min_p10_all = float(np.min(p10_band))

        if expected_return_pct >= 0:
            take_profit = round(max(max_p90_all, max_pred_high, last_real_close * 1.008), 4)
            stop_loss = round(min(min_p10_all, min_pred_low, last_real_close * 0.992), 4)
        else:
            take_profit = round(min(min_p10_all, min_pred_low, last_real_close * 0.992), 4)
            stop_loss = round(max(max_p90_all, max_pred_high, last_real_close * 1.008), 4)

        return {
            "model_used": self.model_id,
            "params_label": "24.7M params" if model_size == "small" else "4.1M params",
            "context_len": len(work_df),
            "device": self.device,
            "sample_count": sample_count,
            "paths_up": paths_up,
            "p_up": round(p_up, 2),
            "regime": regime,
            "last_close": round(last_real_close, 4),
            "predicted_final_close": pred_final_close,
            "p90_final": round(max_p90_all, 4),
            "p10_final": round(min_p10_all, 4),
            "expected_return_pct": expected_return_pct,
            "bull_prob_pct": bull_prob_pct,
            "max_predicted_high": round(max_pred_high, 4),
            "min_predicted_low": round(min_pred_low, 4),
            "volatility_range_pct": volatility_range_pct,
            "suggested_take_profit": take_profit,
            "suggested_stop_loss": stop_loss,
            "predicted_candles": pred_candles,
            "monte_carlo_paths": monte_carlo_paths,
        }

    def backtest_recent_windows_batch(
        self,
        df: pd.DataFrame,
        num_windows: int = 8,
        horizon: int = 4,
        samples_per_win: int = 10,
        ctx_len: int = 96,
    ) -> Dict[str, Any]:
        """
        Ejecuta la Red Neuronal Kronos REAL en lote paralelo sobre las últimas `num_windows`
        ventanas históricas de `horizon` velas cada una, y además la combina con el filtro
        de flujo institucional (mechas + volumen) para comparar:
        1. Equity Kronos Puro
        2. Equity Híbrido (Kronos + Filtro Anti-Trampas)
        """
        if len(df) < (ctx_len + num_windows * horizon + 4):
            return {"logs": [], "equity": 10000.0, "raw_equity": 10000.0, "trades": 0, "wins": 0, "raw_trades": 0, "raw_wins": 0}

        price_cols = ["open", "high", "low", "close"]
        vol_col = "volume"
        amt_col = "amount"

        work_df = df.copy().reset_index(drop=True)
        if vol_col not in work_df.columns:
            work_df[vol_col] = 0.0
        if amt_col not in work_df.columns:
            work_df[amt_col] = work_df[vol_col] * work_df[price_cols].mean(axis=1)

        all_time_df = calc_time_stamps(work_df["timestamps"])
        raw_vals = work_df[price_cols + [vol_col, amt_col]].values.astype(np.float32)
        stamp_vals = all_time_df.values.astype(np.float32)

        # Índices donde empieza cada ventana de evaluación de 4 velas
        end_limit = len(work_df) - horizon
        start_indices = [end_limit - (num_windows - 1 - i) * horizon for i in range(num_windows)]

        x_batch_list = []
        x_stamp_list = []
        y_stamp_list = []
        means_close = []
        stds_close = []
        last_closes = []

        for e in start_indices:
            win_x = raw_vals[e - ctx_len : e]
            win_x_stamp = stamp_vals[e - ctx_len : e]
            win_y_stamp = stamp_vals[e : e + horizon]

            m = np.mean(win_x, axis=0)
            s = np.std(win_x, axis=0)
            norm_x = np.clip((win_x - m) / (s + 1e-5), -self.clip, self.clip)

            x_batch_list.append(norm_x)
            x_stamp_list.append(win_x_stamp)
            y_stamp_list.append(win_y_stamp)
            means_close.append(float(m[3]))
            stds_close.append(float(s[3]))
            last_closes.append(float(win_x[-1, 3]))

        # Apilar en un solo tensor (num_windows, ctx_len, 6) y repetir por samples_per_win
        x_np = np.stack(x_batch_list, axis=0)
        xs_np = np.stack(x_stamp_list, axis=0)
        ys_np = np.stack(y_stamp_list, axis=0)

        with torch.no_grad():
            x_t = torch.from_numpy(x_np).to(self.device)
            xs_t = torch.from_numpy(xs_np).to(self.device)
            ys_t = torch.from_numpy(ys_np).to(self.device)

            # Repetir cada ventana `samples_per_win` veces: total batch = num_windows * samples_per_win
            B = x_t.size(0)
            x_rep = x_t.unsqueeze(1).repeat(1, samples_per_win, 1, 1).reshape(B * samples_per_win, ctx_len, 6)
            xs_rep = xs_t.unsqueeze(1).repeat(1, samples_per_win, 1, 1).reshape(B * samples_per_win, ctx_len, 5)
            ys_rep = ys_t.unsqueeze(1).repeat(1, samples_per_win, 1, 1).reshape(B * samples_per_win, horizon, 5)

            x_token = self.tokenizer.encode(x_rep, half=True)
            full_stamp = torch.cat([xs_rep, ys_rep], dim=1)

            pre_seq = x_token[0]
            post_seq = x_token[1]

            for i in range(horizon):
                cur_len = pre_seq.size(1)
                cur_stamp = full_stamp[:, :cur_len, :].contiguous()
                s1_logits, context = self.model.decode_s1(pre_seq, post_seq, cur_stamp)
                sample_pre = sample_from_logits(s1_logits[:, -1, :], temperature=1.0, top_k=0, top_p=0.90, sample_logits=True)
                s2_logits = self.model.decode_s2(context, sample_pre)
                sample_post = sample_from_logits(s2_logits[:, -1, :], temperature=1.0, top_k=0, top_p=0.90, sample_logits=True)

                pre_seq = torch.cat([pre_seq, sample_pre], dim=1)
                post_seq = torch.cat([post_seq, sample_post], dim=1)

            z = self.tokenizer.decode([pre_seq, post_seq], half=True)
            # Tomar el cierre predicho al final del horizonte (vela e + horizon - 1)
            z_final_close = z[:, -1, 3].reshape(B, samples_per_win).cpu().numpy()

        logs = []
        hybrid_equity = 10000.0
        raw_equity = 10000.0
        hybrid_trades = 0
        hybrid_wins = 0
        raw_trades = 0
        raw_wins = 0

        for idx, e in enumerate(start_indices):
            pred_closes = z_final_close[idx] * (stds_close[idx] + 1e-5) + means_close[idx]
            prev_c = last_closes[idx]
            p_up = float(np.mean(pred_closes > prev_c))

            hh = pd.Timestamp(work_df["timestamps"].iloc[e]).strftime("%d %H:%M")

            # Analizar el filtro de estructura/flujo en la vela previa a entrar (e-1)
            prev_bar = work_df.iloc[e - 1]
            rng = max(float(prev_bar["high"]) - float(prev_bar["low"]), 1e-8)
            upper_wick = (float(prev_bar["high"]) - max(float(prev_bar["open"]), float(prev_bar["close"]))) / rng
            lower_wick = (min(float(prev_bar["open"]), float(prev_bar["close"])) - float(prev_bar["low"])) / rng

            # EMA rápida para tendencia local
            ema_slice = work_df["close"].iloc[max(0, e - 20) : e].mean()
            trend_up = prev_c >= ema_slice

            entry_open = float(work_df["open"].iloc[e])
            exit_close = float(work_df["close"].iloc[e + horizon - 1])

            # 1. Evaluación Kronos Puro (regla run_kronos.py: p_up >= 0.65 LONG, p_up <= 0.35 SHORT)
            if 0.35 < p_up < 0.65:
                logs.append({
                    "status": "skip",
                    "time_label": hh,
                    "side": "CHOPPY",
                    "p_up": f"{p_up:.2f}",
                    "ret_pct": 0.0,
                    "pnl": 0.0,
                    "filter_note": "Kronos neutral",
                })
                continue

            raw_side = 1 if p_up >= 0.65 else -1
            raw_r = raw_side * ((exit_close / max(entry_open, 1e-8)) - 1.0)
            raw_pnl = 10000.0 * raw_r - 8.0
            raw_equity += raw_pnl
            raw_trades += 1
            if raw_pnl > 0:
                raw_wins += 1

            # 2. Evaluación Filtro Híbrido (Bloquea LONG si hay fuerte mecha de rechazo superior o contra-tendencia extrema, y viceversa)
            blocked_by_flow = False
            if raw_side == 1 and (upper_wick > 0.52 or (not trend_up and lower_wick < 0.15)):
                blocked_by_flow = True
            elif raw_side == -1 and (lower_wick > 0.52 or (trend_up and upper_wick < 0.15)):
                blocked_by_flow = True

            if blocked_by_flow:
                logs.append({
                    "status": "skip",
                    "time_label": hh,
                    "side": "FILTER",
                    "p_up": f"{p_up:.2f}",
                    "ret_pct": round(raw_r * 100.0, 2),
                    "pnl": 0.0,
                    "filter_note": "Bloqueado por Flujo",
                })
                continue

            # Si ambos motores (Kronos + Flujo) están alineados, ejecuta el trade híbrido
            hybrid_equity += raw_pnl
            hybrid_trades += 1
            is_win = raw_pnl > 0
            if is_win:
                hybrid_wins += 1

            logs.append({
                "status": "WIN" if is_win else "LOSS",
                "time_label": hh,
                "side": "LONG" if raw_side == 1 else "SHORT",
                "p_up": f"{p_up:.2f}",
                "ret_pct": round(raw_r * 100.0, 2),
                "pnl": round(raw_pnl, 2),
                "filter_note": "Confirmado 2/2",
            })

        return {
            "logs": logs,
            "equity": round(hybrid_equity, 2),
            "trades": hybrid_trades,
            "wins": hybrid_wins,
            "raw_equity": round(raw_equity, 2),
            "raw_trades": raw_trades,
            "raw_wins": raw_wins,
        }


kronos_engine = KronosEngine()
