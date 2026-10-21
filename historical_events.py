"""
Módulo de Memoria Histórica (Fecha + Vela + Noticia)
Liga movimientos históricos reales del mercado con la noticia que los provocó
y detecta patrones repetidos ("Gemelos Históricos") comparando:
1. Estructura de las velas (mechas de absorción, pico de volumen, volatilidad).
2. Contexto de sentimiento (Pánico Geopolítico, Euforia ETF, FED, Capitulación).
3. Dirección proyectada por Kronos.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


HISTORICAL_EVENTS_DB: List[Dict[str, Any]] = [
    {
        "id": "btc_etf_2024_01_10",
        "date": "2024-01-10",
        "asset": "BTC / Cripto",
        "title": "Aprobación Oficial de los ETF de Bitcoin Spot en EE.UU. (SEC)",
        "category": "Euforia / Sell the News",
        "news_sentiment": 92,        # Euforia extrema en prensa y X
        "fear_greed": 76,            # Codicia alta
        "initial_move_pct": 4.8,     # Mechazo alcista inicial a $49,000
        "volume_spike_ratio": 2.4,   # 2.4x el volumen medio
        "upper_wick_ratio": 0.45,    # Mecha superior larga (venta institucional arriba)
        "lower_wick_ratio": 0.15,
        "aftermath_48h_pct": -7.5,
        "aftermath_14d_pct": -18.2,  # Caída de $49k a $38.5k ("Sell the news")
        "aftermath_60d_pct": 68.0,   # Tras limpiar apalancados, subió a $73k
        "pattern_type": "SELL_THE_NEWS",
        "lesson": (
            "Clásico 'Compra el rumor, vende la noticia'. Las noticias eran inmejorables y "
            "las redes sociales estaban en euforia total, pero las velas dejaron mechas superiores "
            "de distribución institucional. Cayó un -18% durante 12 días para limpiar a la masa "
            "antes de iniciar la verdadera subida hacia $73,000."
        ),
    },
    {
        "id": "iran_israel_2024_04_13",
        "date": "2024-04-13",
        "asset": "BTC / Mercados Globales",
        "title": "Escalada Bélica en Oriente Medio (Ataque con drones y misiles)",
        "category": "Guerra / Geopolítica (Buy the Panic)",
        "news_sentiment": -88,       # Pánico global en titulares
        "fear_greed": 34,            # Miedo repentino
        "initial_move_pct": -8.9,    # Desplome rápido de $67k a $60.5k
        "volume_spike_ratio": 2.9,   # Fuerte pico de volumen en la caída
        "upper_wick_ratio": 0.12,
        "lower_wick_ratio": 0.58,    # Gran mecha inferior de absorción institucional
        "aftermath_48h_pct": 7.8,    # Rebote inmediato a $66k
        "aftermath_14d_pct": 9.5,
        "aftermath_60d_pct": 14.0,
        "pattern_type": "BUY_THE_PANIC",
        "lesson": (
            "Pánico geopolítico en titulares que provocó ventas masivas de minoristas y barrido de "
            "stop-losses. Sin embargo, el volumen récord con mecha inferior larga mostró absorción "
            "institucional ('Smart Money' comprando el miedo), recuperando toda la caída en 48h."
        ),
    },
    {
        "id": "ukraine_war_2022_02_24",
        "date": "2022-02-24",
        "asset": "BTC / S&P 500 / Nasdaq",
        "title": "Inicio de la Guerra en Ucrania (Invasión de madrugada)",
        "category": "Guerra / Geopolítica (Wall of Worry)",
        "news_sentiment": -95,       # Pánico mundial extremo
        "fear_greed": 20,            # Miedo extremo
        "initial_move_pct": -9.2,    # Caída brusca a $34,300
        "volume_spike_ratio": 3.4,   # Volumen institucional masivo
        "upper_wick_ratio": 0.10,
        "lower_wick_ratio": 0.65,    # Mecha de absorción histórica en vela diaria
        "aftermath_48h_pct": 14.5,   # Cerró en positivo el mismo día y subió fuerte
        "aftermath_14d_pct": 24.0,   # Subida hasta $44,000 en días posteriores
        "aftermath_60d_pct": 18.0,
        "pattern_type": "BUY_THE_PANIC",
        "lesson": (
            "El día que las noticias eran más aterradoras a nivel mundial, el mercado marcó el suelo "
            "exacto a primera hora. El dinero institucional absorbió toda la liquidez de pánico "
            "dejando una enorme mecha inferior y subiendo más de un +20% en la semana siguiente."
        ),
    },
    {
        "id": "yen_carry_crash_2024_08_05",
        "date": "2024-08-05",
        "asset": "BTC / Bolsa Mundial (Nikkei & Nasdaq)",
        "title": "Crash del Yen Japonés y Pánico de Recesión en EE.UU. ('Lunes Negro')",
        "category": "Capitulación Macroeconómica",
        "news_sentiment": -90,
        "fear_greed": 17,            # Miedo extremo (<20)
        "initial_move_pct": -12.4,   # Caída vertical hasta $49,000
        "volume_spike_ratio": 3.8,   # Mayor volumen del trimestre
        "upper_wick_ratio": 0.08,
        "lower_wick_ratio": 0.62,    # Mecha inferior gigante de absorción
        "aftermath_48h_pct": 15.2,   # Rebote explosivo a $57,000
        "aftermath_14d_pct": 22.0,
        "aftermath_60d_pct": 35.0,
        "pattern_type": "BUY_THE_PANIC",
        "lesson": (
            "Capitulación de manual: cuando el índice Fear & Greed cae por debajo de 20 y aparece "
            "una vela con volumen >3x y mecha inferior >50%, estadísticamente marca el agotamiento "
            "de los vendedores y el inicio de acumulación institucional."
        ),
    },
    {
        "id": "fed_pivot_2024_09_18",
        "date": "2024-09-18",
        "asset": "BTC / S&P 500 / Oro",
        "title": "Recorte de Tipos de Interés de la Reserva Federal (FED -50 bps)",
        "category": "Política Monetaria / Liquidez Alcista",
        "news_sentiment": 65,        # Positivo moderado (sin euforia ciega)
        "fear_greed": 52,            # Neutral-Alcista
        "initial_move_pct": 3.2,
        "volume_spike_ratio": 1.8,
        "upper_wick_ratio": 0.18,
        "lower_wick_ratio": 0.25,
        "aftermath_48h_pct": 5.4,
        "aftermath_14d_pct": 11.2,
        "aftermath_60d_pct": 48.0,
        "pattern_type": "TREND_CONTINUATION",
        "lesson": (
            "Cuando una noticia macroeconómica de inyección de liquidez ocurre con el sentimiento "
            "en zona neutral (sin euforia minorista previa) y velas de cuerpo sólido, la tendencia "
            "alcista tiene recorrido real y sostenido."
        ),
    },
    {
        "id": "svb_banking_crisis_2023_03_11",
        "date": "2023-03-11",
        "asset": "BTC / Sector Financiero",
        "title": "Quiebra de Silicon Valley Bank (SVB) y Pánico en Stablecoins",
        "category": "Pánico Sistémico / Giro Alcista",
        "news_sentiment": -85,
        "fear_greed": 33,
        "initial_move_pct": -6.5,    # Caída a $19,600
        "volume_spike_ratio": 2.6,
        "upper_wick_ratio": 0.14,
        "lower_wick_ratio": 0.52,
        "aftermath_48h_pct": 21.0,   # Subida explosiva a $24,200
        "aftermath_14d_pct": 42.0,   # Rally a $28,000
        "aftermath_60d_pct": 46.0,
        "pattern_type": "BUY_THE_PANIC",
        "lesson": (
            "Mientras las noticias hablaban de colapso bancario, las velas mostraron absorción "
            "en soporte clave y giro en 'V' impulsado por capital buscando refugio."
        ),
    },
    {
        "id": "distribution_top_2024_03_14",
        "date": "2024-03-14",
        "asset": "BTC / Cripto",
        "title": "Euforia Minorista en Máximos ($73,700) y Récord de Flujos en Portadas",
        "category": "Euforia Extrema / Techo Local",
        "news_sentiment": 88,
        "fear_greed": 88,            # Codicia extrema (>80)
        "initial_move_pct": 2.1,
        "volume_spike_ratio": 2.1,
        "upper_wick_ratio": 0.54,    # Rechazo fuerte en parte alta
        "lower_wick_ratio": 0.12,
        "aftermath_48h_pct": -9.2,
        "aftermath_14d_pct": -15.4,
        "aftermath_60d_pct": -12.0,
        "pattern_type": "SELL_THE_NEWS",
        "lesson": (
            "Con el índice de Codicia en 88 y todos los medios publicando titulares triunfalistas, "
            "las velas mostraron agotamiento (cuerpos pequeños y mechas de rechazo por arriba). "
            "Fue señal clara de toma de beneficios institucional."
        ),
    },
    {
        "id": "real_breakdown_2022_05_09",
        "date": "2022-05-09",
        "asset": "BTC / Cripto",
        "title": "Ruptura de Soporte sin Absorción (Colapso de Liquidez LUNA/UST)",
        "category": "Caída Real Sin Absorción (Cuchillo Cayendo)",
        "news_sentiment": -82,
        "fear_greed": 15,
        "initial_move_pct": -10.5,
        "volume_spike_ratio": 2.8,
        "upper_wick_ratio": 0.15,
        "lower_wick_ratio": 0.08,    # ¡OJO! Vela llena cerrada en mínimos (SIN mecha de compra)
        "aftermath_48h_pct": -14.0,
        "aftermath_14d_pct": -18.5,
        "aftermath_60d_pct": -36.0,
        "pattern_type": "REAL_BREAKDOWN",
        "lesson": (
            "Diferencia vital: cuando hay malas noticias pero la vela cierra en el mínimo absoluto "
            "SIN mecha inferior de absorción y Kronos proyecta continuación bajista, NO se compra "
            "el pánico porque las instituciones aún no están comprando."
        ),
    },
    {
        "id": "xrp_sec_ruling_2023_07_13",
        "date": "2023-07-13",
        "asset": "XRP / Cripto",
        "title": "Fallo Judicial a favor de Ripple (XRP): Las ventas en exchanges NO son valores (SEC)",
        "category": "Catalizador Regulatorio / Explosión de Liquidez",
        "news_sentiment": 94,
        "fear_greed": 64,
        "initial_move_pct": 73.0,    # Subida de $0.47 a $0.82 en horas
        "volume_spike_ratio": 4.5,   # Volumen institucional récord
        "upper_wick_ratio": 0.38,    # Mecha superior en $0.93 tras el primer impulso
        "lower_wick_ratio": 0.10,
        "aftermath_48h_pct": -11.5,  # Toma de beneficios tras vela vertical (+73%)
        "aftermath_14d_pct": -14.0,
        "aftermath_60d_pct": -32.0,
        "pattern_type": "SELL_THE_NEWS",
        "lesson": (
            "En XRP, cuando una noticia regulatoria dispara el precio más de un +50% en una sola sesión "
            "dejando una mecha superior pronunciada (>35%), históricamente los creadores de mercado "
            "toman beneficios en las siguientes 48h–14 días."
        ),
    },
    {
        "id": "xrp_institutional_breakout_2024_11_16",
        "date": "2024-11-16",
        "asset": "XRP / Cripto",
        "title": "Ruptura de Acumulación Multi-Anual de XRP y Giro Regulatorio en EE.UU.",
        "category": "Acumulación Institucional / Tendencia Fuerte",
        "news_sentiment": 72,
        "fear_greed": 74,
        "initial_move_pct": 18.5,
        "volume_spike_ratio": 3.2,
        "upper_wick_ratio": 0.16,
        "lower_wick_ratio": 0.28,
        "aftermath_48h_pct": 24.0,
        "aftermath_14d_pct": 145.0,  # Rally histórico de $0.80 a $2.70+
        "aftermath_60d_pct": 210.0,
        "pattern_type": "TREND_CONTINUATION",
        "lesson": (
            "Cuando XRP rompe un rango largo con volumen >3x y las velas cierran cerca del máximo "
            "(mecha superior pequeña <20%), no es una trampa sino el inicio de una expansión "
            "institucional de varias semanas."
        ),
    },
]


def extract_candle_signature(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extrae la firma matemática de las últimas velas reales para compararla
    con eventos históricos (mechas de absorción, pico de volumen, retorno reciente).
    """
    if len(df) < 20:
        return {
            "recent_move_pct": 0.0,
            "volume_spike_ratio": 1.0,
            "upper_wick_ratio": 0.2,
            "lower_wick_ratio": 0.2,
            "absorption_score": 0.0,
            "distribution_score": 0.0,
        }

    recent = df.iloc[-5:]
    last = df.iloc[-1]
    prev_close = float(df.iloc[-6]["close"]) if len(df) >= 6 else float(df.iloc[0]["close"])
    last_close = float(last["close"])
    recent_move_pct = ((last_close - prev_close) / max(prev_close, 1e-8)) * 100.0

    avg_vol = float(df.iloc[-25:-1]["volume"].mean()) if "volume" in df.columns else 1.0
    max_recent_vol = float(recent["volume"].max()) if "volume" in df.columns else 1.0
    volume_spike_ratio = max_recent_vol / max(avg_vol, 1e-8)

    # Analizar las mechas de las últimas 3 velas para detectar absorción o distribución
    last3 = df.iloc[-3:]
    upper_ratios = []
    lower_ratios = []
    for _, row in last3.iterrows():
        high_v = float(row["high"])
        low_v = float(row["low"])
        open_v = float(row["open"])
        close_v = float(row["close"])
        rng = max(high_v - low_v, 1e-8)
        body_top = max(open_v, close_v)
        body_bot = min(open_v, close_v)
        upper_ratios.append((high_v - body_top) / rng)
        lower_ratios.append((body_bot - low_v) / rng)

    upper_wick_ratio = float(np.mean(upper_ratios))
    lower_wick_ratio = float(np.mean(lower_ratios))

    # Puntuación de absorción institucional (mechas inferiores largas + volumen alto)
    absorption_score = min(100.0, (lower_wick_ratio * 110.0) * min(volume_spike_ratio, 2.5))
    # Puntuación de distribución institucional (mechas superiores largas + volumen alto)
    distribution_score = min(100.0, (upper_wick_ratio * 110.0) * min(volume_spike_ratio, 2.5))

    return {
        "recent_move_pct": round(recent_move_pct, 2),
        "volume_spike_ratio": round(volume_spike_ratio, 2),
        "upper_wick_ratio": round(upper_wick_ratio, 2),
        "lower_wick_ratio": round(lower_wick_ratio, 2),
        "absorption_score": round(absorption_score, 1),
        "distribution_score": round(distribution_score, 1),
    }


def find_historical_twins(
    candle_sig: Dict[str, float],
    news_sentiment: float,
    fear_greed: float,
    kronos_expected_return_pct: float,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """
    Compara la situación actual (Velas + Noticias + Fear&Greed + Proyección Kronos)
    con la base de datos de eventos históricos para detectar patrones repetidos.
    """
    results = []

    for ev in HISTORICAL_EVENTS_DB:
        # 1. Similitud en mechas (absorción vs distribución)
        wick_diff = (
            abs(candle_sig["lower_wick_ratio"] - ev["lower_wick_ratio"]) +
            abs(candle_sig["upper_wick_ratio"] - ev["upper_wick_ratio"])
        ) / 2.0
        wick_sim = max(0.0, 1.0 - wick_diff * 1.4)

        # 2. Similitud en sentimiento y Fear & Greed
        sent_diff = abs(news_sentiment - ev["news_sentiment"]) / 200.0
        fg_diff = abs(fear_greed - ev["fear_greed"]) / 100.0
        psych_sim = max(0.0, 1.0 - (sent_diff * 0.6 + fg_diff * 0.4))

        # 3. Similitud en volumen relativo
        vol_diff = min(1.0, abs(candle_sig["volume_spike_ratio"] - ev["volume_spike_ratio"]) / 3.0)
        vol_sim = max(0.0, 1.0 - vol_diff)

        # 4. Alineación con la dirección proyectada por Kronos y el desenlace histórico (48h)
        kronos_bullish = kronos_expected_return_pct > 0.15
        kronos_bearish = kronos_expected_return_pct < -0.15
        ev_bullish = ev["aftermath_48h_pct"] > 0
        if (kronos_bullish and ev_bullish) or (kronos_bearish and not ev_bullish):
            direction_bonus = 0.15
        else:
            direction_bonus = 0.0

        raw_score = (wick_sim * 0.35 + psych_sim * 0.35 + vol_sim * 0.15 + direction_bonus)
        similarity_pct = round(min(97.0, max(42.0, raw_score * 100.0)), 1)

        ev_copy = dict(ev)
        ev_copy["similarity_pct"] = similarity_pct
        results.append(ev_copy)

    results.sort(key=lambda x: x["similarity_pct"], reverse=True)
    return results[:top_n]
