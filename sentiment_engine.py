"""
Motor de Noticias en Vivo + Índice Fear & Greed + Detector de Trampas de Masas
Conecta a internet para leer titulares financieros en tiempo real, calcula el sentimiento
y detecta divergencias entre lo que dice la prensa/masa y lo que hacen las velas institucionales.
"""

from typing import Dict, Any, List
import time
import urllib.parse
import requests
import feedparser


BULLISH_KEYWORDS = [
    "surge", "rally", "soar", "bull", "breakout", "inflow", "record", "high",
    "adoption", "approval", "etf", "accumulation", "whale", "buy", "cut rates",
    "stimulus", "upgrade", "gain", "optimism", "beat", "sube", "alcista", "récord",
    "aprobación", "acumulación", "rebote", "compras", "máximos",
]

BEARISH_KEYWORDS = [
    "crash", "plunge", "drop", "bear", "hack", "exploit", "lawsuit", "ban",
    "outflow", "liquidation", "panic", "recession", "inflation", "hawkish",
    "selloff", "dump", "bankruptcy", "default", "fear", "cae", "desplome",
    "pánico", "liquidaciones", "inflación", "ventas", "riesgo", "crisis",
]

GEOPOLITICAL_KEYWORDS = [
    "war", "missile", "strike", "military", "conflict", "invasion", "sanctions",
    "middle east", "ukraine", "iran", "israel", "russia", "taiwan", "tariff",
    "geopolitical", "escalation", "tension", "guerra", "ataque", "conflicto",
    "misiles", "geopolítica", "sanciones", "aranceles",
]

MACRO_FED_KEYWORDS = [
    "fed", "powell", "fomc", "interest rate", "cpi", "inflation", "jobs",
    "treasury", "central bank", "ecb", "gdp", "yield", "tipos de interés",
    "reserva federal", "banco central",
]


ASSET_SEARCH_QUERIES = {
    "BTCUSDT": "Bitcoin OR Crypto market",
    "ETHUSDT": "Ethereum OR ETH crypto",
    "XRPUSDT": "XRP OR Ripple crypto SEC",
    "SOLUSDT": "Solana SOL crypto",
    "BNBUSDT": "BNB Binance crypto",
    "PAXGUSDT": "Gold price XAU geopolitical",
    "^GSPC": "S&P 500 stock market Wall Street Fed",
    "^IXIC": "Nasdaq tech stocks Wall Street",
    "GC=F": "Gold price commodities safe haven",
    "NVDA": "Nvidia NVDA AI stock",
}


def fetch_fear_and_greed() -> Dict[str, Any]:
    """Obtiene el Índice Oficial de Miedo y Codicia (Fear & Greed Index) en vivo."""
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=2", timeout=5)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                val = int(data[0].get("value", 50))
                classification = data[0].get("value_classification", "Neutral")
                prev_val = int(data[1].get("value", val)) if len(data) > 1 else val
                return {
                    "value": val,
                    "previous": prev_val,
                    "classification": classification,
                    "label_es": translate_fg_label(val),
                    "source": "alternative.me (En vivo)",
                }
    except Exception:
        pass

    return {
        "value": 50,
        "previous": 50,
        "classification": "Neutral",
        "label_es": "Neutral (50/100)",
        "source": "Fallback Local",
    }


def translate_fg_label(val: int) -> str:
    if val <= 22:
        return "Miedo Extremo (Zona de Capitulación)"
    if val <= 42:
        return "Miedo (Cautela en la Masa)"
    if val <= 58:
        return "Neutral (Equilibrio)"
    if val <= 78:
        return "Codicia (Optimismo Minorista)"
    return "Codicia Extrema (Alerta de Euforia / Techo)"


def score_headline(title: str) -> Dict[str, Any]:
    text = title.lower()
    bull_hits = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
    bear_hits = sum(1 for kw in BEARISH_KEYWORDS if kw in text)
    geo_hits = sum(1 for kw in GEOPOLITICAL_KEYWORDS if kw in text)
    macro_hits = sum(1 for kw in MACRO_FED_KEYWORDS if kw in text)

    # Si es noticia de guerra/geopolítica suele causar miedo inmediato en titulares
    raw_score = (bull_hits * 32) - (bear_hits * 32) - (geo_hits * 25)
    score = max(-100, min(100, raw_score))

    if geo_hits > 0:
        tag = "🌍 Geopolítica / Conflicto"
    elif macro_hits > 0:
        tag = "🏦 Macro / Reserva Federal"
    elif score > 15:
        tag = "🟢 Alcista / Euforia"
    elif score < -15:
        tag = "🔴 Bajista / Temor"
    else:
        tag = "⚪ Mercado / Neutral"

    return {
        "score": score,
        "tag": tag,
        "is_geopolitical": geo_hits > 0,
        "is_macro": macro_hits > 0,
    }


def fetch_live_news(symbol: str, max_items: int = 12) -> Dict[str, Any]:
    """
    Descarga noticias reales en vivo desde Google News RSS + CoinDesk/Cointelegraph/Yahoo
    para el activo seleccionado y analiza su sentimiento.
    """
    query = ASSET_SEARCH_QUERIES.get(symbol, f"{symbol} financial market")
    encoded_q = urllib.parse.quote(query)

    rss_urls = [
        f"https://news.google.com/rss/search?q={encoded_q}+when:2d&hl=en-US&gl=US&ceid=US:en",
    ]
    if symbol.endswith("USDT"):
        rss_urls.append("https://www.coindesk.com/arc/outboundfeeds/rss/")
    else:
        rss_urls.append("https://finance.yahoo.com/news/rssindex")

    articles: List[Dict[str, Any]] = []
    seen_titles = set()

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    for url in rss_urls:
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code != 200:
                continue
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:10]:
                title = (entry.get("title") or "").strip()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                analysis = score_headline(title)
                published = entry.get("published", "Reciente")
                link = entry.get("link", "#")
                source_title = "Prensa Financiera"
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    title = parts[0].strip()
                    source_title = parts[1].strip()

                articles.append({
                    "title": title,
                    "source": source_title,
                    "published": published[:25] if published else "Hoy",
                    "link": link,
                    "score": analysis["score"],
                    "tag": analysis["tag"],
                    "is_geopolitical": analysis["is_geopolitical"],
                    "is_macro": analysis["is_macro"],
                })
                if len(articles) >= max_items:
                    break
        except Exception:
            continue
        if len(articles) >= max_items:
            break

    # Calcular sentimiento agregado de las noticias
    if articles:
        non_zero = [a["score"] for a in articles if a["score"] != 0]
        if non_zero:
            avg_news_score = float(sum(non_zero)) / len(non_zero)
        else:
            avg_news_score = 0.0
        geo_count = sum(1 for a in articles if a["is_geopolitical"])
    else:
        avg_news_score = 0.0
        geo_count = 0

    return {
        "articles": articles,
        "news_sentiment_score": round(avg_news_score, 1),
        "geopolitical_alert_count": geo_count,
        "timestamp": int(time.time()),
    }


def evaluate_crowd_trap_and_signal(
    kronos_return_pct: float,
    kronos_bull_prob_pct: float,
    candle_sig: Dict[str, float],
    news_score: float,
    fear_greed_val: int,
    geo_alert_count: int,
    tv_score: float,
) -> Dict[str, Any]:
    """
    Cerebro Híbrido: Combina la proyección de Kronos + Indicadores TradingView +
    Sentimiento de Noticias + Fear & Greed + Detector de Trampas de la Masa.
    """
    # Normalizar sentimiento de la masa combinando noticias (-100..+100) y Fear&Greed (0..100 -> -100..+100)
    fg_centered = (fear_greed_val - 50) * 2.0
    crowd_sentiment = round(news_score * 0.55 + fg_centered * 0.45, 1)

    absorption = candle_sig.get("absorption_score", 0.0)
    distribution = candle_sig.get("distribution_score", 0.0)

    trap_detected = False
    trap_type = "NONE"
    trap_title = "Sin Trampa de Masa Detectada (Mercado Normal)"
    trap_explanation = (
        "El sentimiento de las noticias y la acción del precio en las velas no muestran "
        "manipulación extrema en este momento."
    )

    # CASO 1: Pánico Geopolítico / Miedo Extremo + Absorción Institucional ("Buy the Panic / Wall of Worry")
    if (crowd_sentiment < -22 or fear_greed_val <= 32 or geo_alert_count >= 2) and (
        kronos_return_pct > 0.10 or absorption >= 35.0
    ):
        trap_detected = True
        trap_type = "BUY_THE_PANIC"
        trap_title = "🎯 DIVERGENCIA ALCISTA: Pánico en Noticias pero Absorción en Velas ('Buy the Panic')"
        trap_explanation = (
            f"Las noticias/masa muestran miedo (Sentimiento {crowd_sentiment:+.0f}, Fear & Greed {fear_greed_val}/100), "
            f"pero las velas muestran mechas de compra institucional (Absorción {absorption:.0f}/100) y "
            f"Kronos proyecta subida ({kronos_return_pct:+.2f}%). Históricamente (como en Ucrania 2022 o Oriente Medio 2024), "
            f"cuando el dinero fuerte absorbe el pánico de las noticias, el mercado rebota al alza."
        )

    # CASO 2: Euforia Extrema en Noticias + Distribución / Agotamiento en Kronos ("Sell the News")
    elif (crowd_sentiment > 28 or fear_greed_val >= 72) and (
        kronos_return_pct < 0.05 or distribution >= 38.0
    ):
        trap_detected = True
        trap_type = "SELL_THE_NEWS"
        trap_title = "⚠️ TRAMPA DE EUFORIA: 'Sell the News' (Distribución Institucional Detectada)"
        trap_explanation = (
            f"La masa y los titulares están eufóricos (Sentimiento {crowd_sentiment:+.0f}, Fear & Greed {fear_greed_val}/100), "
            f"pero las velas muestran rechazo superior (Distribución {distribution:.0f}/100) y Kronos proyecta "
            f"agotamiento ({kronos_return_pct:+.2f}%). Peligro de que las instituciones estén vendiendo sus posiciones a los minoristas."
        )

    # CASO 3: Noticia Negativa + Ruptura Real Sin Absorción ("Cuchillo Cayendo")
    elif crowd_sentiment < -25 and kronos_return_pct < -0.25 and absorption < 25.0:
        trap_detected = True
        trap_type = "REAL_BREAKDOWN"
        trap_title = "🛑 CAÍDA REAL SIN ABSORCIÓN: No comprar el pánico todavía"
        trap_explanation = (
            "Las noticias son negativas y, a diferencia de un suelo institucional, las velas aún NO dejan "
            "mechas de compra y Kronos proyecta continuación bajista. Espera a que aparezca volumen de absorción."
        )

    # Calcular la señal final ponderando los 3 motores + ajuste contrarian
    # 1. Voto Kronos (-100 a +100)
    kronos_vote = max(-100.0, min(100.0, kronos_return_pct * 38.0 + (kronos_bull_prob_pct - 50.0) * 1.1))
    # 2. Voto Técnico TradingView (-100 a +100)
    tech_vote = tv_score
    # 3. Voto Psicológico / Contrarian
    if trap_type == "BUY_THE_PANIC":
        psych_vote = 65.0  # Impulso comprador por absorción de pánico
    elif trap_type == "SELL_THE_NEWS":
        psych_vote = -70.0  # Bloqueo/venta por trampa de euforia
    else:
        psych_vote = max(-50.0, min(50.0, crowd_sentiment * 0.5))

    final_score = round(kronos_vote * 0.55 + tech_vote * 0.25 + psych_vote * 0.20, 1)
    confidence = round(min(96.0, max(51.0, 50.0 + abs(final_score) * 0.52)), 1)

    if final_score >= 42:
        action = "COMPRA FUERTE (LONG)"
        color = "emerald"
    elif final_score >= 14:
        action = "COMPRA (LONG)"
        color = "green"
    elif final_score <= -42:
        action = "VENTA FUERTE (SHORT)"
        color = "rose"
    elif final_score <= -14:
        action = "VENTA / TOMAR BENEFICIOS"
        color = "red"
    else:
        action = "ESPERAR / NEUTRAL"
        color = "amber"

    return {
        "action": action,
        "color": color,
        "final_score": final_score,
        "confidence_pct": confidence,
        "crowd_sentiment": crowd_sentiment,
        "kronos_vote": round(kronos_vote, 1),
        "tech_vote": round(tech_vote, 1),
        "psych_vote": round(psych_vote, 1),
        "trap_detected": trap_detected,
        "trap_type": trap_type,
        "trap_title": trap_title,
        "trap_explanation": trap_explanation,
    }
