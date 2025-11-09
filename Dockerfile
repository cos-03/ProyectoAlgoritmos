# Dockerfile - ejemplo para Railway
FROM python:3.12-slim

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copiar requirements y código
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias (puede tardar)
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copiar todo el repo
COPY . /app

# Crear directorio para outputs
RUN mkdir -p /app/outputs

# Puerto que Railway expone (Railway usa $PORT en runtime)
ENV PORT 8080

# Comando por defecto: uvicorn
CMD ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
