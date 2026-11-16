FROM ghcr.io/astral-sh/uv:0.13.8 AS uv
FROM python:3.12-slim-bookworm

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /srv/delivery-risk

ENV PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY app/ ./app/

USER 10001:10001
EXPOSE 8000
CMD [".venv/bin/python", "-m", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
