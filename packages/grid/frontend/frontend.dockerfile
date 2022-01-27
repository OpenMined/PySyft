ARG FRONTEND_DEV
ARG DISABLE_TELEMETRY=1
ARG PRODUCTION_DIR=/prod_app

FROM node:16-alpine as init-stage
ARG PRODUCTION_DIR
ARG DISABLE_TELEMETRY

ENV NODE_TYPE $NODE_TYPE
ENV PROD_ROOT $PRODUCTION_DIR
ENV NEXT_TELEMETRY_DISABLED $DISABLE_TELEMETRY
ENV NEXT_PUBLIC_API_URL=/api/v1

WORKDIR /app
COPY package.json yarn.lock /app/
RUN --mount=type=cache,target=/root/.yarn YARN_CACHE_FOLDER=/root/.yarn yarn --frozen-lockfile
COPY . .

FROM node:16-alpine as grid-ui-development
ARG DISABLE_TELEMETRY

ENV NODE_TYPE $NODE_TYPE
ENV NEXT_TELEMETRY_DISABLED $DISABLE_TELEMETRY
ENV NEXT_PUBLIC_ENVIRONMENT=development
ENV NEXT_PUBLIC_API_URL=/api/v1

WORKDIR /app
COPY --from=init-stage /app .
CMD ["/usr/local/bin/node", "--max-old-space-size=4096", "/app/node_modules/.bin/next", "dev", "-p", "80"]

FROM init-stage as build-stage
WORKDIR $PROD_ROOT

COPY --from=init-stage /app .
RUN yarn build
RUN yarn export

FROM nginx:stable-alpine as grid-ui-production
ARG DISABLE_TELEMETRY
ARG PRODUCTION_DIR

ENV NEXT_TELEMETRY_DISABLED $DISABLE_TELEMETRY
ENV PROD_ROOT $PRODUCTION_DIR
ENV NEXT_PUBLIC_ENVIRONMENT=production
ENV NEXT_PUBLIC_API_URL=/api/v1
ENV NODE_TYPE $NODE_TYPE

COPY --from=build-stage $PROD_ROOT/out /usr/share/nginx/html
COPY --from=build-stage $PROD_ROOT/docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build-stage $PROD_ROOT/docker/nginx-backend-not-found.conf /etc/nginx/extra-conf.d/backend-not-found
COPY --from=build-stage $PROD_ROOT /hauuuh

