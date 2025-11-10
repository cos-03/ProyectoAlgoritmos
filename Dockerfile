# Dockerfile (recomendado para Render)
FROM python:3.12-slim

# Evitar prompts apt
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# copia requisitos (puedes usar requirements.txt o requirements_light.txt)
COPY requirements.txt /app/requirements.txt

# pequeños ajustes para builds reproducibles
RUN apt-get update \
  && apt-get install -y build-essential curl git libglib2.0-0 libnss3 libgconf-2-4 libxss1 libxtst6 \
  && pip install --upgrade pip setuptools wheel \
  && pip install -r /app/requirements.txt \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# copia todo el proyecto
COPY . /app

# directorio de salida
RUN mkdir -p /app/outputs

# Exponer puerto (nota: Render sobrescribirá con $PORT)
EXPOSE 8000

# CMD usando la variable PORT (Render exporta $PORT)
CMD ["sh", "-lc", "uvicorn main_fastapi:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 120"]
