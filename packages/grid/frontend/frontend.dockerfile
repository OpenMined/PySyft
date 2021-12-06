FROM node:16-alpine as init-stage

WORKDIR /app
COPY package.json yarn.lock /app/
RUN --mount=type=cache,target=/root/.yarn YARN_CACHE_FOLDER=/root/.yarn yarn --frozen-lockfile
COPY . .

FROM node:16-alpine as grid-ui-development

ENV NEXT_PUBLIC_ENVIRONMENT development
ENV NEXT_PUBLIC_API_URL /api/v1
ENV NODE_TYPE $NODE_TYPE

WORKDIR /app
COPY --from=init-stage /app .
CMD ["/usr/local/bin/node", "--max-old-space-size=4096", "/app/node_modules/.bin/next", "dev", "-p", "80"]

FROM init-stage as build-stage
ARG NODE_TYPE
WORKDIR /app
COPY --from=init-stage /app .
ENV NODE_TYPE $NODE_TYPE
ENV NEXT_TELEMETRY_DISABLED 1
RUN yarn build
RUN yarn export

FROM nginx:stable-alpine as grid-ui-production

ENV NEXT_PUBLIC_ENVIRONMENT production
ENV NEXT_PUBLIC_API_URL /api/v1
ENV NODE_TYPE $NODE_TYPE

COPY --from=build-stage /app/out /usr/share/nginx/html
COPY --from=build-stage /app/docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build-stage /app/docker/nginx-backend-not-found.conf /etc/nginx/extra-conf.d/backend-not-found
