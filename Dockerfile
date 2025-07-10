FROM python:3.12-slim AS server

WORKDIR /app

RUN chmod 1777 /tmp

RUN apt-get update \
      && apt-get install -y build-essential \
      && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY server/ .

EXPOSE 8000

CMD ["python", "main.py"]
