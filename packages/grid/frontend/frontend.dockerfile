ARG TYPE
ARG FRONTEND_DEV
ARG NEXT_TELEMETRY_DISABLED=1
ARG PRODUCTION_DIR=/prod_app

FROM node:16-alpine as init-stage
ARG PRODUCTION_DIR
ARG NEXT_TELEMETRY_DISABLED
ARG TYPE

ENV PROD_ROOT $PRODUCTION_DIR
ENV NODE_TYPE $TYPE

WORKDIR /app
COPY package.json yarn.lock .
RUN yarn --frozen-lockfile
COPY . .

FROM init-stage as build-stage
WORKDIR $PROD_ROOT

COPY --from=init-stage /app .
RUN yarn build
RUN yarn export

FROM nginx:stable-alpine as grid-ui-production
ARG NEXT_TELEMETRY_DISABLED=1
ARG PRODUCTION_DIR

ENV PROD_ROOT $PRODUCTION_DIR

COPY --from=build-stage $PROD_ROOT/out /usr/share/nginx/html
COPY --from=build-stage $PROD_ROOT/docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build-stage $PROD_ROOT/docker/nginx-backend-not-found.conf /etc/nginx/extra-conf.d/backend-not-found
COPY --from=build-stage $PROD_ROOT /hauuuh

FROM node:16-alpine as grid-ui-development
ARG NEXT_TELEMETRY_DISABLED
ARG TYPE

ENV NODE_TYPE $TYPE

WORKDIR /app
COPY --from=init-stage /app .
CMD ["/usr/local/bin/node", "--max-old-space-size=4096", "/app/node_modules/.bin/next", "dev", "-p", "80"]
