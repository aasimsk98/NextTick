FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
RUN pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

COPY . .

ENV NEXTTICK_MODELS_DIR=/tmp/nexttick-models
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:create_app()"]