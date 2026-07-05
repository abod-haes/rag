FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=180
ENV PIP_RETRIES=10

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 180 --retries 10 -r requirements.txt

COPY ./app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
