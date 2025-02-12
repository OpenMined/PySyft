FROM cgr.dev/chainguard/wolfi-base

ARG PYTHON_VERSION="3.12"
ARG UV_VERSION="0.4.20-r0"
ARG SYFT_VERSION="0.3.5"

RUN apk update && apk upgrade && \
    apk add --no-cache python-$PYTHON_VERSION uv=$UV_VERSION

WORKDIR /app

RUN uv venv
RUN uv pip install --no-cache syftbox==${SYFT_VERSION}

EXPOSE 8000

CMD ["uv", "run", "gunicorn", "syftbox.server.server:app", "--bind=0.0.0.0:8000"]
