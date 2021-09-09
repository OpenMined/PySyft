ARG TYPE
ARG FRONTEND_DEV
ARG DISABLE_TELEMETRY=1
ARG PRODUCTION_DIR=/prod_app

FROM node:16-alpine as init-stage
ARG TYPE
ARG PRODUCTION_DIR
ARG DISABLE_TELEMETRY

ENV NODE_TYPE $TYPE
ENV PROD_ROOT $PRODUCTION_DIR
ENV NEXT_TELEMETRY_DISABLED $DISABLE_TELEMETRY

WORKDIR /app
COPY package.json yarn.lock .
RUN yarn --frozen-lockfile
COPY . .

FROM node:16-alpine as grid-ui-development
ARG TYPE
ARG DISABLE_TELEMETRY

ENV NODE_TYPE $TYPE
ENV NEXT_TELEMETRY_DISABLED $DISABLE_TELEMETRY

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

COPY --from=build-stage $PROD_ROOT/out /usr/share/nginx/html
COPY --from=build-stage $PROD_ROOT/docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build-stage $PROD_ROOT/docker/nginx-backend-not-found.conf /etc/nginx/extra-conf.d/backend-not-found
COPY --from=build-stage $PROD_ROOT /hauuuh


