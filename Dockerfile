FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --system app && useradd --system --gid app --home-dir /app app
WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN python -m pip install .

COPY --chown=app:app configs ./configs
COPY --chown=app:app models ./models

USER app
EXPOSE 8000

CMD ["uvicorn", "faturaiz.api:app", "--host", "0.0.0.0", "--port", "8000"]

