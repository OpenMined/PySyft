ARG PYTHON_VERSION="3.12"
FROM python:${PYTHON_VERSION}-bookworm
RUN apt update && apt install -y netcat-openbsd vim
WORKDIR /app
CMD ["python3", "-m", "http.server", "8000"]
EXPOSE 8000

# docker build -f domain.dockerfile . -t domain
# docker run -it -p 8080:8000 domain
