$env:HAGRID_VERSION = $(python hagrid/version.py)
docker buildx build -f hagrid.dockerfile -t openmined/hagrid:"$env:HAGRID_VERSION" -t openmined/hagrid:latest .