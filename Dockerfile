FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir '.[server]' && useradd --create-home worker \
    && mkdir /work && chown worker:worker /work
USER worker
EXPOSE 8765
# Pass --config /config/api.yaml --allow-network explicitly for model-backed serving.
ENTRYPOINT ["streambudget"]
CMD ["serve", "--out", "/work/session"]
