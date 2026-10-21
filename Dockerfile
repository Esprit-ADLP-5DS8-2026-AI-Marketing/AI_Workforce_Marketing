# syntax=docker/dockerfile:1.7
FROM python:3.10-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/opt/huggingface \
    RAG_DEVICE=cpu RAG_OFFLINE=true TOKENIZERS_PARALLELISM=false
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir "torch==2.10.0+cpu" \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cpu \
 && python -m pip install --no-cache-dir -r /tmp/requirements-docker.txt \
 && python -m pip check
COPY scripts/ /app/scripts/
COPY prompts/ /app/prompts/
COPY data/raw/ /app/data/raw/
COPY docker/ /app/docker/
RUN python docker/prepare_image.py \
 && python -m pip freeze > /app/docker/installed-packages.txt \
 && mkdir -p /app/data/products /app/data/answers
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
 CMD ["python", "docker/healthcheck.py"]
CMD ["python", "scripts/run_rag_api.py", "--host", "0.0.0.0", "--port", "8000"]

# Explicitly requested private image: only these four settings are embedded.
# Pass .env via --secret; it is excluded from the ordinary build context.
FROM runtime AS configured
RUN --mount=type=secret,id=rag_env,required=true \
    python docker/embed_config.py /run/secrets/rag_env /app/.env
