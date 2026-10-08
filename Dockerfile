FROM python:3.10.6-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /bin/

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    STREAMLIT_SERVER_HEADLESS=true

COPY pyproject.toml .python-version ./
RUN uv sync --no-dev

COPY . .

EXPOSE 10000
CMD ["sh", "-c", "uv run streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-10000}"]
