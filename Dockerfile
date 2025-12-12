FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git-lfs \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV LAST_UPDATED="2023-12-12-15301" 
ENV PYTHONUNBUFFERED=1

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN chmod +x /app/docker_startup.sh

CMD ["/app/docker_startup.sh"]
