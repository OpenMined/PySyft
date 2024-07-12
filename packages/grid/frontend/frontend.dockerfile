FROM cgr.dev/chainguard/wolfi-base as base

ARG BACKEND_API_BASE_URL="/api/v2/"
ENV BACKEND_API_BASE_URL ${BACKEND_API_BASE_URL}

RUN apk update && \
  apk upgrade && \
  apk add --no-cache nodejs-20 pnpm corepack

ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"

WORKDIR /app

RUN corepack enable

COPY .npmrc ./
COPY package.json ./
COPY pnpm-lock.yaml ./

FROM base AS dependencies

RUN --mount=type=cache,id=pnpm,target=/pnpm/store pnpm install --frozen-lockfile

FROM dependencies as syft-ui-tests
COPY vite.config.ts ./
COPY ./tests ./tests
COPY ./src/ ./src

CMD pnpm test:unit

FROM dependencies as syft-ui-development

ENV SERVER_ENV=development

COPY . .
CMD pnpm dev

FROM dependencies AS builder

COPY . .
RUN pnpm build

FROM base AS syft-ui-production

ENV SERVER_ENV=production

COPY --from=dependencies /app/node_modules ./node_modules
COPY --from=builder /app ./
CMD pnpm preview
