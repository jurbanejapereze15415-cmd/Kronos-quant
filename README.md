---
title: Kronos Quant Terminal
emoji: 📈
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# KRONOS QUANT — Terminal Institucional 24/7 en la Nube (Hugging Face Spaces)

Terminal cuantitativa híbrida conectada a **`shiyu-coder/Kronos`**, **Binance API**, **Yahoo Finance**, **Lector de Noticias en Tiempo Real** y **Puente iOS (iPhone)**.

## Cómo subir este bot a Hugging Face Spaces (100% Gratis y con tu Mac apagado)

1. Entra en **[huggingface.co/spaces](https://huggingface.co/spaces)** e inicia sesión (o crea una cuenta gratuita).
2. Pulsa el botón **"Create new Space"** (arriba a la derecha).
3. Rellena los 3 datos básicos:
   - **Space name:** `kronos-quant` (o el nombre que quieras).
   - **Select the Space SDK:** Elige **`Docker`** (y selecciona la plantilla **Blank**).
   - **Space Hardware:** Déjalo en **`CPU basic · 2 vCPU · 16 GB · FREE`**.
4. Pulsa **"Create Space"**.
5. En la página que se abre, haz clic en la pestaña **"Files"** (arriba a la derecha) -> **"Add file"** -> **"Upload files"**.
6. Arrastra desde tu carpeta `/Users/joanurbaneja/.gemini/antigravity/scratch/kronos-trading-bot` estos **9 archivos sueltos** (¡no hace falta ni siquiera que subas las carpetas `Kronos` ni `.venv` porque el `Dockerfile` descarga Kronos automáticamente!):
   - `Dockerfile`
   - `README.md`
   - `requirements.txt`
   - `app.py`
   - `index.html`
   - `kronos_service.py`
   - `market_data.py`
   - `sentiment_engine.py`
   - `historical_events.py`
   - `iphone_bridge.py`
7. Pulsa **"Commit changes to main"**.
8. Hugging Face construirá el servidor automáticamente en ~2 minutos y arriba te pondrá **`Running`**.
9. Para abrirlo en tu iPhone a pantalla completa con el Mac apagado, abre la dirección **`https://TU-USUARIO-kronos-quant.hf.space`** en Safari de tu iPhone y pulsa **Compartir -> Añadir a pantalla de inicio**.
