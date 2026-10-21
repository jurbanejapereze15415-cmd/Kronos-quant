FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN if [ ! -d "Kronos/model" ]; then git clone https://github.com/shiyu-coder/Kronos.git Kronos; fi

ENV PORT=10000
EXPOSE $PORT

CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"
