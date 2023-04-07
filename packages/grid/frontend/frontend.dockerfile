FROM node:18-alpine as base

ENV NODE_TYPE domain

WORKDIR /app
COPY .npmrc package.json pnpm-lock.yaml ./

# cant use the cache for multi architecture builds in CI because it fails
# https://github.com/docker/buildx/issues/549
# RUN --mount=type=cache,target=/root/.yarn YARN_CACHE_FOLDER=/root/.yarn yarn install --frozen-lockfile
RUN npm i -g pnpm

# Stage 2: Install dependencies
FROM base AS dependencies
COPY pnpm-lock.yaml package.json ./
RUN pnpm install --frozen-lockfile

FROM dependencies as grid-ui-development
ENV NODE_ENV=development
CMD ["pnpm", "dev"]

# Stage 3: Build the Svelte project
FROM dependencies AS builder
COPY . .
RUN pnpm run build


# Stage 4: Production image
FROM base AS grid-ui-production
COPY --from=dependencies /app/node_modules ./node_modules
COPY --from=builder /app ./
COPY pnpm-lock.yaml package.json ./

# Set the environment to production mode
ENV NODE_ENV=production

# Start the production server
CMD ["pnpm", "preview"]