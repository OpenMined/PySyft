#### INSTRUCTIONS
# BUILD the syft base image first

# for DEV mode, editable source and hot reloading
# $ docker build -f docker/grid.Dockerfile --build-arg APP=domain --build-arg APP_ENV=dev -t openmined/grid-domain-dev:latest -t openmined/grid-domain-dev:`python VERSION` .
# $ docker run -it -v "`pwd`/packages/grid/apps/domain:/app" -v "`pwd`/packages/syft:/syft" -p 5000:5000 openmined/grid-domain-dev

# for PROD mode, non editable and smaller
# $ docker build -f docker/grid.Dockerfile --build-arg APP=domain --build-arg APP_ENV=production --build-arg VERSION=`python VERSION` -t openmined/grid-domain:latest -t openmined/grid-domain:`python VERSION` .
# $ docker run -it -p 5000:5000 openmined/grid-domain

# for PROD mode, non editable and smaller
# $ docker build -f docker/grid.Dockerfile --build-arg APP=network --build-arg APP_ENV=production --build-arg VERSION=`python VERSION` -t openmined/grid-network:latest -t openmined/grid-network:`python VERSION` .
# $ docker run -it -p 5000:5000 openmined/grid-network

ARG VERSION=latest
FROM openmined/syft:$VERSION

# envs and args
ARG APP
ARG APP_ENV=production
ENV APP_ENV=$APP_ENV
ENV DATABASE_URL=sqlite:///nodedatabase.db
ENV PORT=5000

RUN --mount=type=cache,target=/root/.cache python3 -m pip install poetry

# copy and setup app
RUN mkdir -p /app
COPY ./packages/grid/apps/$APP /app
RUN cd /app && poetry export -f requirements.txt --output requirements.txt --without-hashes
RUN --mount=type=cache,target=/root/.cache pip install -r /app/requirements.txt
RUN pip install psycopg2-binary

# run the app
CMD ["bash", "-c", "cd /app && ./run.sh"]
