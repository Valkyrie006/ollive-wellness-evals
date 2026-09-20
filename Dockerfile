# syntax=docker/dockerfile:1
#
# Two stages so the CUDA-free CPU torch wheels are resolved once and the
# final image carries no build toolchain. The CPU index matters: the default
# torch wheel drags in several GB of NVIDIA CUDA libraries that are useless
# for sentence-transformers inference on a CPU box.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch first, so the resolver doesn't pull the CUDA build later.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    # Cache the embedding model inside the image layer rather than
    # re-downloading it on every container start.
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY . .

# Bake the embedding model in, so a cold start doesn't depend on
# huggingface.co being reachable and fast.
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('all-MiniLM-L6-v2')" \
    && chmod -R a+rX /opt/hf-cache

# Run unprivileged: a container that only needs to read its own code has no
# business running as root.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
