FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY eval/ ./eval/

ENTRYPOINT ["python", "-m", "eval.critique"]
