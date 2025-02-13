# Etapa de construcción
FROM python:3.12.4 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Crear un entorno virtual en /app/.venv
RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install --no-cache-dir -r requirements.txt

# Etapa final (imagen ligera)
FROM python:3.12.4-slim
WORKDIR /app

# Copiar el entorno virtual y el código fuente
COPY --from=builder /app/.venv .venv/
COPY . .

# Definir el comando de ejecución correcto para Flask
CMD [".venv/bin/python", "app.py"]
