# Dockerfile para desplegar FastAPI con Playwright y PyTorch (optimizado)
FROM python:3.12-slim

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Directorio de trabajo
WORKDIR /app

# Instalar dependencias de sistema necesarias para playwright y otras libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    ca-certificates \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (mejor caché Docker)
COPY requirements.txt /app/requirements.txt

# Instalar python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Instalar navegadores de playwright
RUN python -m playwright install --with-deps

# Copiar el código
COPY . /app

# Exponer puerto interno (por convención 8000)
EXPOSE 8000

# CMD por defecto (ajusta el module si tu app está en otro archivo)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main_fastapi:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
