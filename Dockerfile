FROM python:3.10-slim

WORKDIR /app

# System deps (ffmpeg for AAC/M4A streaming)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências
COPY requirements-silero.txt .
RUN pip install --no-cache-dir -r requirements-silero.txt

# Copia toda a raiz do projeto (inclusive main.py, utils.py, .env etc)
COPY . .

EXPOSE 8000

# Usa o main.py diretamente
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# uvicorn main:app --host 0.0.0.0 --port 8000
