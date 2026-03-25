FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/cache data/models

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "btcdump.web.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
