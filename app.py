"""
Servidor Principal del Bot Híbrido Kronos AI + Conexión iPhone
Conecta en tiempo real:
1. Motor IA Fundacional Kronos (shiyu-coder/Kronos) con Abanico Monte Carlo
2. Binance API (Cripto) & Yahoo Finance (Índices, Oro, Acciones)
3. Indicadores Técnicos estilo TradingView (RSI, MACD, EMAs, Bollinger)
4. Lector de Noticias en Vivo + Índice Fear & Greed + Detector de Trampas de Masas
5. Buscador de Gemelos Históricos (Fecha + Vela + Noticia)
6. Puente iPhone (WebApp iOS por Wi-Fi + Notificaciones Push ntfy.sh y Telegram)
"""

import os
from typing import Dict, Any
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

from market_data import (
    SUPPORTED_ASSETS,
    fetch_market_klines,
    make_future_timestamps,
    compute_tradingview_indicators,
)
from kronos_service import kronos_engine
from sentiment_engine import (
    fetch_fear_and_greed,
    fetch_live_news,
    evaluate_crowd_trap_and_signal,
)
from historical_events import (
    HISTORICAL_EVENTS_DB,
    extract_candle_signature,
    find_historical_twins,
)
from iphone_bridge import (
    get_local_ip,
    get_public_or_local_url,
    load_iphone_config,
    save_iphone_config,
    send_push_to_iphone,
)

app = FastAPI(title="Kronos Quant — Institutional Terminal & iPhone Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def simulate_paper_trading_log(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Genera el registro de Paper Trading estilo `run_kronos.py` sobre las últimas ventanas
    del gráfico actual:
    - Si 0.35 < p_up < 0.65 -> skip (CHOPPY)
    - Si p_up >= 0.65 -> LONG | Si p_up <= 0.35 -> SHORT
    - PnL sobre $10,000 de capital con $8 de comisión
    """
    logs = []
    equity = 10000.0
    trades_count = 0
    wins_count = 0

    if len(df) < 80:
        return {"logs": [], "equity": 10000.0, "trades": 0, "wins": 0}

    # Evaluar las últimas 10 ventanas de 4 velas cada una
    step = 4
    indices = list(range(len(df) - 44, len(df) - 4, step))
    for e in indices[-8:]:
        win = df.iloc[max(0, e - 30) : e]
        future_slice = df.iloc[e : e + 4]
        if len(win) < 10 or len(future_slice) < 4:
            continue

        # Estimación estadística rápida de p_up en esa ventana histórica
        ret_recent = (float(win["close"].iloc[-1]) - float(win["close"].iloc[-6])) / max(float(win["close"].iloc[-6]), 1e-8)
        lower_wick = (min(float(win["open"].iloc[-1]), float(win["close"].iloc[-1])) - float(win["low"].iloc[-1]))
        rng = max(float(win["high"].iloc[-1]) - float(win["low"].iloc[-1]), 1e-8)
        wick_bias = (lower_wick / rng) - 0.5
        p_up = float(np.clip(0.5 + ret_recent * 8.5 + wick_bias * 0.35, 0.10, 0.90))
        p_up = round(p_up * 20.0) / 20.0  # Redondear a múltiplos de 0.05 como en run_kronos.py

        hh = pd.Timestamp(df["timestamps"].iloc[e]).strftime("%d %H:%M")

        if 0.35 < p_up < 0.65:
            logs.append({
                "status": "skip",
                "time_label": hh,
                "side": "FLAT",
                "p_up": f"{p_up:.2f}",
                "ret_pct": 0.0,
                "pnl": 0.0,
            })
            continue

        side = 1 if p_up >= 0.65 else -1
        side_str = "LONG" if side == 1 else "SHORT"
        entry_open = float(df["open"].iloc[e])
        exit_close = float(df["close"].iloc[e + 3])
        r = side * ((exit_close / max(entry_open, 1e-8)) - 1.0)
        pnl = 10000.0 * r - 8.0
        equity += pnl
        trades_count += 1
        is_win = pnl > 0
        if is_win:
            wins_count += 1

        logs.append({
            "status": "WIN" if is_win else "LOSS",
            "time_label": hh,
            "side": side_str,
            "p_up": f"{p_up:.2f}",
            "ret_pct": round(r * 100.0, 2),
            "pnl": round(pnl, 2),
        })

    return {
        "logs": logs,
        "equity": round(equity, 2),
        "trades": trades_count,
        "wins": wins_count,
    }


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/iphone")
def get_iphone_connection_info():
    ip = get_local_ip()
    cfg = load_iphone_config()
    return {
        "local_ip": ip,
        "iphone_url": get_public_or_local_url(),
        "config": cfg,
    }


@app.post("/api/iphone/config")
def update_iphone_config(payload: Dict[str, Any] = Body(...)):
    cfg = load_iphone_config()
    if "ntfy_topic" in payload:
        cfg["ntfy_topic"] = str(payload["ntfy_topic"]).strip()
    if "telegram_bot_token" in payload:
        cfg["telegram_bot_token"] = str(payload["telegram_bot_token"]).strip()
    if "telegram_chat_id" in payload:
        cfg["telegram_chat_id"] = str(payload["telegram_chat_id"]).strip()
    if "auto_push_on_signal" in payload:
        cfg["auto_push_on_signal"] = bool(payload["auto_push_on_signal"])
    save_iphone_config(cfg)
    return {"status": "ok", "config": cfg}


@app.post("/api/iphone/push")
def trigger_iphone_push(payload: Dict[str, Any] = Body(...)):
    res = send_push_to_iphone(payload)
    return res


@app.get("/api/assets")
def list_assets():
    return {"assets": SUPPORTED_ASSETS, "historical_events": HISTORICAL_EVENTS_DB}


@app.get("/api/analyze")
def analyze_market(
    symbol: str = Query("BTCUSDT", description="Símbolo del activo"),
    interval: str = Query("1h", description="Temporalidad: 5m, 15m, 1h, 4h, 1d"),
    pred_len: int = Query(24, ge=6, le=48, description="Velas futuras a predecir"),
    model_size: str = Query("mini", description="Tamaño del modelo Kronos: mini o small"),
    backtest_mode: bool = Query(False, description="Si es True, oculta las últimas velas reales para comprobar acierto"),
):
    try:
        df_full, data_source = fetch_market_klines(symbol=symbol, interval=interval, limit=360)
        if len(df_full) < 60:
            raise ValueError("Datos históricos insuficientes para analizar este activo.")

        actual_hidden_candles = []
        backtest_metrics = None

        if backtest_mode and len(df_full) > (pred_len + 60):
            df_input = df_full.iloc[:-pred_len].copy().reset_index(drop=True)
            df_hidden = df_full.iloc[-pred_len:].copy().reset_index(drop=True)
            y_ts = df_hidden["timestamps"]
            for _, row in df_hidden.iterrows():
                actual_hidden_candles.append({
                    "time": int(pd.Timestamp(row["timestamps"]).timestamp()),
                    "open": round(float(row["open"]), 4),
                    "high": round(float(row["high"]), 4),
                    "low": round(float(row["low"]), 4),
                    "close": round(float(row["close"]), 4),
                })
        else:
            df_input = df_full.copy().reset_index(drop=True)
            last_ts = pd.Timestamp(df_input["timestamps"].iloc[-1])
            y_ts = make_future_timestamps(last_ts, interval=interval, pred_len=pred_len)

        # 1. Ejecutar predicción probabilística con Kronos (24 futuros Monte Carlo)
        kronos_res = kronos_engine.predict_multi_scenarios(
            df=df_input,
            x_timestamp=df_input["timestamps"],
            y_timestamp=y_ts,
            pred_len=pred_len,
            sample_count=24,
            temperature=1.0,
            top_p=0.92,
            model_size=model_size,
        )

        if backtest_mode and actual_hidden_candles:
            start_price = kronos_res["last_close"]
            real_end_price = actual_hidden_candles[-1]["close"]
            pred_end_price = kronos_res["predicted_final_close"]
            real_return_pct = round(((real_end_price - start_price) / max(start_price, 1e-8)) * 100.0, 2)
            pred_return_pct = kronos_res["expected_return_pct"]
            direction_hit = (real_return_pct >= 0 and pred_return_pct >= 0) or (
                real_return_pct < 0 and pred_return_pct < 0
            )
            mae_pct = round(abs(real_return_pct - pred_return_pct), 2)
            backtest_metrics = {
                "enabled": True,
                "start_price": start_price,
                "real_end_price": real_end_price,
                "pred_end_price": pred_end_price,
                "real_return_pct": real_return_pct,
                "pred_return_pct": pred_return_pct,
                "direction_hit": bool(direction_hit),
                "error_pct": mae_pct,
            }

        # 2. Calcular indicadores estilo TradingView
        tv_indicators = compute_tradingview_indicators(df_input)

        # 3. Extraer firma de las velas
        candle_sig = extract_candle_signature(df_input)

        # 4. Consultar Noticias en Vivo + Índice Fear & Greed
        fg_data = fetch_fear_and_greed()
        news_data = fetch_live_news(symbol=symbol, max_items=10)

        # 5. Evaluar Señal Híbrida + Detector de Trampas de la Masa
        signal_eval = evaluate_crowd_trap_and_signal(
            kronos_return_pct=kronos_res["expected_return_pct"],
            kronos_bull_prob_pct=kronos_res["bull_prob_pct"],
            candle_sig=candle_sig,
            news_score=news_data["news_sentiment_score"],
            fear_greed_val=fg_data["value"],
            geo_alert_count=news_data["geopolitical_alert_count"],
            tv_score=tv_indicators["tv_score"],
        )

        # 6. Buscar Gemelos Históricos (Fecha + Vela + Noticia)
        historical_twins = find_historical_twins(
            candle_sig=candle_sig,
            news_sentiment=news_data["news_sentiment_score"],
            fear_greed=fg_data["value"],
            kronos_expected_return_pct=kronos_res["expected_return_pct"],
            top_n=3,
        )

        # 7. Simulador Paper Trading Real en Lote (Kronos Neuronal + Filtro Híbrido en paralelo)
        paper_sim = kronos_engine.backtest_recent_windows_batch(df_input)

        # 8. Si el usuario activó auto-push al iPhone, enviar notificación
        cfg = load_iphone_config()
        if cfg.get("auto_push_on_signal"):
            send_push_to_iphone({
                "symbol": symbol,
                "interval": interval,
                "action": signal_eval["action"],
                "price": kronos_res["last_close"],
                "pred_close": kronos_res["predicted_final_close"],
                "ret_pct": kronos_res["expected_return_pct"],
                "tp": kronos_res["suggested_take_profit"],
                "sl": kronos_res["suggested_stop_loss"],
                "regime": kronos_res["regime"],
                "p_up": kronos_res["p_up"],
                "trap_title": signal_eval["trap_title"],
            })

        historical_candles = []
        for _, row in df_input.tail(250).iterrows():
            historical_candles.append({
                "time": int(pd.Timestamp(row["timestamps"]).timestamp()),
                "open": round(float(row["open"]), 4),
                "high": round(float(row["high"]), 4),
                "low": round(float(row["low"]), 4),
                "close": round(float(row["close"]), 4),
                "volume": round(float(row["volume"]), 2),
            })

        return JSONResponse({
            "symbol": symbol,
            "interval": interval,
            "data_source": data_source,
            "historical_candles": historical_candles,
            "actual_hidden_candles": actual_hidden_candles,
            "backtest_metrics": backtest_metrics,
            "kronos": kronos_res,
            "tradingview": tv_indicators,
            "candle_signature": candle_sig,
            "fear_greed": fg_data,
            "news": news_data,
            "signal": signal_eval,
            "historical_twins": historical_twins,
            "paper_sim": paper_sim,
            "iphone": {
                "local_ip": get_local_ip(),
                "iphone_url": get_public_or_local_url(),
                "ntfy_topic": cfg.get("ntfy_topic"),
            },
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
