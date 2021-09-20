$env:HAGRID_VERSION = $(python hagrid/__init__.py)
docker buildx build -f hagrid.dockerfile -t openmined/hagrid:"$env:HAGRID_VERSION" -t openmined/hagrid:latest .