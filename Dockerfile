# Simple development image for credit risk project
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app \
	MLFLOW_TRACKING_URI=file:/app/mlruns \
	MODEL_NAME=credit-risk-probability-model \
	MODEL_STAGE=Production

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
