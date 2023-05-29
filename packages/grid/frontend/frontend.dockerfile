FROM node:18-alpine as base

ARG VITE_PUBLIC_API_BASE_URL
ENV VITE_PUBLIC_API_BASE_URL ${VITE_PUBLIC_API_BASE_URL}
ENV NODE_TYPE domain

WORKDIR /app

RUN corepack enable && corepack prepare pnpm@latest --activate

COPY .npmrc ./
COPY package.json ./
COPY pnpm-lock.yaml ./

FROM base AS dependencies

RUN pnpm i --frozen-lockfile

FROM dependencies as grid-ui-tests
COPY vite.config.ts ./
COPY ./tests ./tests
COPY ./src/ ./src

CMD pnpm test:unit

FROM dependencies as grid-ui-development

ENV NODE_ENV=development

COPY . .
CMD pnpm dev

FROM dependencies AS builder

COPY . .
RUN pnpm build

FROM base AS grid-ui-production

ENV NODE_ENV=production

COPY --from=dependencies /app/node_modules ./node_modules
COPY --from=builder /app ./
CMD pnpm preview
