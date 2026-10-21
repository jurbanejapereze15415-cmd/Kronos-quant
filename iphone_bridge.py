"""
Puente de Conexión con iPhone (WebApp iOS + Notificaciones Push ntfy & Telegram)
Permite:
1. Conectar el iPhone por Wi-Fi escaneando un código QR e instalar el bot como App en iOS.
2. Enviar alertas Push instantáneas a la pantalla de bloqueo del iPhone vía ntfy (sin registro) o Telegram.
"""

import os
import json
import socket
import random
from typing import Dict, Any
import requests

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iphone_config.json")


def get_local_ip() -> str:
    """Obtiene la IP local Wi-Fi del Mac para que el iPhone pueda conectarse."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "192.168.1.43"


def get_public_or_local_url() -> str:
    """
    Si está desplegado en Hugging Face Spaces (SPACE_HOST), devuelve la URL pública 24/7.
    Si está en local en el Mac, devuelve http://<IP_LOCAL>:8765.
    """
    space_host = os.environ.get("SPACE_HOST", "").strip()
    if space_host:
        return f"https://{space_host}"
    return f"http://{get_local_ip()}:8765"


def load_iphone_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if "ntfy_topic" in cfg:
                    return cfg
        except Exception:
            pass

    default_cfg = {
        "ntfy_topic": f"kronos-quant-iphone-{random.randint(1000, 9999)}",
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "auto_push_on_signal": False,
    }
    save_iphone_config(default_cfg)
    return default_cfg


def save_iphone_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def send_push_to_iphone(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Envía una notificación Push real al iPhone a través de ntfy.sh (y opcionalmente Telegram).
    """
    cfg = load_iphone_config()
    topic = cfg.get("ntfy_topic", "").strip()
    tg_token = cfg.get("telegram_bot_token", "").strip()
    tg_chat = cfg.get("telegram_chat_id", "").strip()

    symbol = payload.get("symbol", "BTCUSDT")
    interval = payload.get("interval", "1h")
    action = payload.get("action", "SEÑAL")
    price = payload.get("price", "--")
    pred_close = payload.get("pred_close", "--")
    ret_pct = payload.get("ret_pct", 0.0)
    tp = payload.get("tp", "--")
    sl = payload.get("sl", "--")
    regime = payload.get("regime", "CHOPPY")
    p_up = payload.get("p_up", 0.5)
    trap_title = payload.get("trap_title", "")

    title = f"KRONOS QUANT · {symbol} ({interval}) -> {action}"
    body_lines = [
        f"Spot: {price} | Mediana Kronos: {pred_close} ({ret_pct:+.2f}%)",
        f"Régimen: {regime} (p_up: {p_up:.2f})",
        f"Take-Profit: {tp} | Stop-Loss: {sl}",
    ]
    if trap_title:
        body_lines.append(f"Flujo: {trap_title}")
    message_body = "\n".join(body_lines)

    results = {"ntfy_sent": False, "telegram_sent": False, "topic": topic}

    # 1. Envío inmediato por ntfy.sh (App gratuita en App Store de iPhone sin cuentas)
    if topic:
        try:
            target_url = get_public_or_local_url()
            headers = {
                "Title": title.encode("utf-8"),
                "Priority": "high",
                "Tags": "chart_with_upwards_trend,iphone",
                "Click": target_url,
            }
            resp = requests.post(
                f"https://ntfy.sh/{topic}",
                data=message_body.encode("utf-8"),
                headers=headers,
                timeout=6,
            )
            results["ntfy_sent"] = resp.status_code == 200
        except Exception as e:
            results["ntfy_error"] = str(e)

    # 2. Envío por Telegram si el usuario configuró Token + Chat ID
    if tg_token and tg_chat:
        try:
            tg_text = f"*{title}*\n\n```text\n{message_body}\n```"
            tg_url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
            resp = requests.post(
                tg_url,
                json={"chat_id": tg_chat, "text": tg_text, "parse_mode": "Markdown"},
                timeout=6,
            )
            results["telegram_sent"] = resp.status_code == 200
        except Exception as e:
            results["telegram_error"] = str(e)

    return results
