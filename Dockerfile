FROM python:3.10-slim-buster

RUN apt-get update -y && apt-get install -y awscli && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]