FROM node:18-alpine as grid-ui-development
ENV NODE_TYPE domain

WORKDIR /app
COPY .npmrc package.json pnpm-lock.yaml ./

# cant use the cache for multi architecture builds in CI because it fails
# https://github.com/docker/buildx/issues/549
# RUN --mount=type=cache,target=/root/.yarn YARN_CACHE_FOLDER=/root/.yarn yarn install --frozen-lockfile
RUN npm i -g pnpm
RUN pnpm install --frozen-lockfile
COPY . .
CMD ["pnpm", "dev"]

FROM grid-ui-development as build-stage
RUN pnpm build

FROM node:18-alpine as grid-ui-production
ENV NODE_TYPE $NODE_TYPE

WORKDIR /app
RUN rm -rf ./*
COPY --from=build-stage /app/package.json .
COPY --from=build-stage /app/build .
RUN pnpm --prod
CMD ["node", "index.js"]
