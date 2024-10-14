FROM k3d-registry.localhost:5800/openmined/syft-backend:dev-latest

RUN uv pip install db-dtypes google-cloud-bigquery