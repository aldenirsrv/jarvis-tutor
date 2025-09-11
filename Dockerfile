FROM python:3.10-slim

WORKDIR /app

# Instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia toda a raiz do projeto (inclusive main.py, utils.py, .env etc)
COPY . .

EXPOSE 8000

# Usa o main.py diretamente
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]