FROM node:16-alpine as init-stage

ENV ENVIRONMENT=${FRONTEND_ENV}

WORKDIR /app

COPY package.json yarn.lock .
RUN yarn --frozen-lockfile

COPY . .

FROM node:16-alpine as grid-ui-development

ENV NEXT_TELEMETRY_DISABLED=1

WORKDIR /app
COPY --from=init-stage /app .
CMD ["/usr/local/bin/node", "--max-old-space-size=4096", "/app/node_modules/.bin/next", "dev", "-p", "80"]

FROM init-stage as build-stage

WORKDIR /app
COPY --from=init-stage /app .

RUN yarn build
RUN yarn export

FROM nginx:stable-alpine as grid-ui-production

ENV NEXT_TELEMETRY_DISABLED=1

COPY --from=build-stage /app/out /usr/share/nginx/html
COPY --from=build-stage /app/docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build-stage /app/docker/nginx-backend-not-found.conf /etc/nginx/extra-conf.d/backend-not-found
