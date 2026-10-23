FROM python:3.11-slim

# Commit the image was built from; reported by /health.
ARG GIT_SHA=local

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/birth_weight_pipeline.joblib \
    PORT=8000 \
    GIT_SHA=${GIT_SHA}

WORKDIR /app

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv

# Install only the serving packages, pinned to the versions in uv.lock, so the
# image runs the same library versions the model was trained with.
COPY pyproject.toml uv.lock requirements-api.txt ./
RUN uv export --frozen --no-hashes --no-emit-project --extra api -o constraints.txt \
    && uv pip install --system --no-cache -r requirements-api.txt -c constraints.txt

COPY src ./src

# The model is part of the image: one image tag = one model version.
COPY models/birth_weight_pipeline.joblib models/model_metadata.json ./models/

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
