FROM python:3.10-slim

# Instalar git para clonar el motor oficial shiyu-coder/Kronos si no está en la carpeta
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root requerido por Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Copiar dependencias e instalarlas (CPU PyTorch ligero para nube gratuita de Hugging Face)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copiar los archivos del proyecto
COPY --chown=user . .

# Si el usuario subió los archivos sueltos sin la carpeta Kronos, clonarla automáticamente
RUN if [ ! -d "Kronos/model" ]; then git clone https://github.com/shiyu-coder/Kronos.git Kronos; fi

# Puerto oficial de Hugging Face Spaces
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
