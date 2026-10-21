"""
Conector de Datos en Tiempo Real (Binance API + Yahoo Finance)
y Calculadora de Indicadores Técnicos estilo TradingView (RSI, MACD, EMAs, Bandas de Bollinger).
"""

from typing import Dict, Any, Tuple
import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf


SUPPORTED_ASSETS = [
    {"symbol": "BTCUSDT", "name": "Bitcoin (BTC/USDT)", "source": "binance", "category": "Cripto"},
    {"symbol": "ETHUSDT", "name": "Ethereum (ETH/USDT)", "source": "binance", "category": "Cripto"},
    {"symbol": "XRPUSDT", "name": "XRP / Ripple (XRP/USDT)", "source": "binance", "category": "Cripto"},
    {"symbol": "SOLUSDT", "name": "Solana (SOL/USDT)", "source": "binance", "category": "Cripto"},
    {"symbol": "BNBUSDT", "name": "BNB (BNB/USDT)", "source": "binance", "category": "Cripto"},
    {"symbol": "PAXGUSDT", "name": "Oro Tokenizado (PAXG/USDT)", "source": "binance", "category": "Materias Primas"},
    {"symbol": "^GSPC", "name": "S&P 500 Index (EE.UU.)", "source": "yahoo", "category": "Índices"},
    {"symbol": "^IXIC", "name": "Nasdaq Composite", "source": "yahoo", "category": "Índices"},
    {"symbol": "GC=F", "name": "Oro Futuros (XAU/USD)", "source": "yahoo", "category": "Materias Primas"},
    {"symbol": "NVDA", "name": "NVIDIA Corp (NVDA)", "source": "yahoo", "category": "Acciones"},
]


INTERVAL_DELTAS = {
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


def fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 320) -> pd.DataFrame:
    """Descarga velas en tiempo real directamente desde la API oficial de Binance."""
    endpoints = [
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines",
    ]
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    last_err = None
    for url in endpoints:
        try:
            resp = requests.get(url, params=params, timeout=7)
            if resp.status_code == 200:
                raw = resp.json()
                rows = []
                for r in raw:
                    rows.append({
                        "timestamps": pd.to_datetime(int(r[0]), unit="ms", utc=True).tz_convert(None),
                        "open": float(r[1]),
                        "high": float(r[2]),
                        "low": float(r[3]),
                        "close": float(r[4]),
                        "volume": float(r[5]),
                        "amount": float(r[7]),  # Quote asset volume
                    })
                df = pd.DataFrame(rows)
                return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Error al conectar con Binance para {symbol}: {last_err}")


def fetch_yahoo_klines(symbol: str, interval: str = "1h", limit: int = 320) -> pd.DataFrame:
    """Descarga velas desde Yahoo Finance para índices, acciones y oro."""
    yf_interval_map = {
        "5m": ("5m", "15d"),
        "15m": ("15m", "30d"),
        "1h": ("1h", "90d"),
        "4h": ("1h", "180d"),
        "1d": ("1d", "2y"),
    }
    yf_int, yf_period = yf_interval_map.get(interval, ("1h", "90d"))
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=yf_period, interval=yf_int)
    if hist is None or hist.empty:
        raise RuntimeError(f"No se pudieron obtener datos de Yahoo Finance para {symbol}")

    hist = hist.reset_index()
    ts_col = "Datetime" if "Datetime" in hist.columns else "Date"
    df = pd.DataFrame({
        "timestamps": pd.to_datetime(hist[ts_col], utc=True).dt.tz_convert(None),
        "open": hist["Open"].astype(float),
        "high": hist["High"].astype(float),
        "low": hist["Low"].astype(float),
        "close": hist["Close"].astype(float),
        "volume": hist["Volume"].fillna(0.0).astype(float),
    })

    if interval == "4h" and yf_int == "1h":
        df = df.set_index("timestamps").resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna().reset_index()

    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    df = df.tail(limit).reset_index(drop=True)
    return df


def fetch_market_klines(symbol: str, interval: str = "1h", limit: int = 320) -> Tuple[pd.DataFrame, str]:
    """Obtiene las velas de Binance o Yahoo Finance según el activo elegido."""
    is_binance = symbol.endswith("USDT")
    if is_binance:
        try:
            df = fetch_binance_klines(symbol, interval=interval, limit=limit)
            return df, "Binance API (Tiempo Real)"
        except Exception:
            # Fallback a Yahoo si Binance falla
            yf_sym = symbol.replace("USDT", "-USD")
            df = fetch_yahoo_klines(yf_sym, interval=interval, limit=limit)
            return df, "Yahoo Finance (Fallback)"
    else:
        df = fetch_yahoo_klines(symbol, interval=interval, limit=limit)
        return df, "Yahoo Finance (Tiempo Real)"


def make_future_timestamps(last_ts: pd.Timestamp, interval: str, pred_len: int) -> pd.Series:
    delta = INTERVAL_DELTAS.get(interval, pd.Timedelta(hours=1))
    future = [last_ts + delta * (i + 1) for i in range(pred_len)]
    return pd.Series(future, name="timestamps")


def compute_tradingview_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula exactamente los indicadores clásicos de TradingView:
    RSI (14), EMA 20, EMA 50, EMA 200, MACD (12, 26, 9), Bandas de Bollinger (20, 2)
    y genera el Velocímetro de Consenso Técnico (-100 a +100).
    """
    close = df["close"].astype(float)
    last_close = float(close.iloc[-1])

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

    # EMAs
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = float((macd_line - signal_line).iloc[-1])

    # Bollinger Bands (20, 2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = float((sma20 + 2 * std20).iloc[-1]) if len(close) >= 20 else last_close * 1.02
    bb_lower = float((sma20 - 2 * std20).iloc[-1]) if len(close) >= 20 else last_close * 0.98

    # Votación estilo velocímetro TradingView
    votes = 0.0
    signals_detail = []

    # 1. Tendencia EMAs
    if last_close > ema20:
        votes += 20
        signals_detail.append("Precio > EMA 20 (Alcista corto plazo)")
    else:
        votes -= 20
        signals_detail.append("Precio < EMA 20 (Debilidad corto plazo)")

    if last_close > ema50:
        votes += 25
        signals_detail.append("Precio > EMA 50 (Estructura alcista)")
    else:
        votes -= 25
        signals_detail.append("Precio < EMA 50 (Estructura bajista)")

    if last_close > ema200:
        votes += 20
        signals_detail.append("Precio > EMA 200 (Tendencia macro alcista)")
    else:
        votes -= 20
        signals_detail.append("Precio < EMA 200 (Tendencia macro bajista)")

    # 2. MACD
    if macd_hist > 0:
        votes += 20
        signals_detail.append("MACD Histograma Positivo (Impulso comprador)")
    else:
        votes -= 20
        signals_detail.append("MACD Histograma Negativo (Impulso vendedor)")

    # 3. RSI (Sobreventa = oportunidad de rebote / Sobrecompra = alerta)
    if rsi < 32:
        votes += 15
        signals_detail.append(f"RSI en Sobreventa ({rsi:.1f}) -> Rebote potencial")
    elif rsi > 72:
        votes -= 15
        signals_detail.append(f"RSI en Sobrecompra ({rsi:.1f}) -> Riesgo de corrección")
    elif rsi >= 52:
        votes += 10
        signals_detail.append(f"RSI en Fuerza Alcista ({rsi:.1f})")
    else:
        votes -= 10
        signals_detail.append(f"RSI Débil ({rsi:.1f})")

    votes = max(-100.0, min(100.0, votes))
    if votes >= 50:
        tv_rating = "COMPRA FUERTE"
    elif votes >= 15:
        tv_rating = "COMPRA"
    elif votes <= -50:
        tv_rating = "VENTA FUERTE"
    elif votes <= -15:
        tv_rating = "VENTA"
    else:
        tv_rating = "NEUTRAL"

    return {
        "rsi": round(rsi, 1),
        "ema20": round(ema20, 4),
        "ema50": round(ema50, 4),
        "ema200": round(ema200, 4),
        "macd_hist": round(macd_hist, 4),
        "bb_upper": round(bb_upper, 4),
        "bb_lower": round(bb_lower, 4),
        "tv_score": round(votes, 1),
        "tv_rating": tv_rating,
        "details": signals_detail,
    }
