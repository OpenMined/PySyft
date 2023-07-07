FROM node:18-alpine as base

WORKDIR /lib

RUN corepack enable && corepack prepare pnpm@latest --activate

COPY rollup.config.js ./
COPY tsconfig.json ./
COPY package.json ./
COPY pnpm-lock.yaml ./

FROM base AS dependencies

RUN pnpm i --frozen-lockfile

FROM dependencies AS syftjs-doc-test
COPY ./src/ ./src

CMD pnpm test:doc

FROM dependencies as syftjs-build
COPY ./src/ ./src

RUN pnpm build

FROM syftjs-build AS syftjs-unit-tests
COPY ./tests ./tests

CMD pnpm test:unit

