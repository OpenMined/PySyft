FROM node:16 as builder

WORKDIR /app
RUN curl -f https://get.pnpm.io/v6.16.js | node - add --global pnpm
COPY package.json pnpm-lock.yaml ./
RUN pnpm i --frozen-lockfile

COPY . .
RUN DOCKER=true pnpm build

FROM node:16-slim
WORKDIR /app
COPY --from=builder /app .
COPY . .

EXPOSE 3000
CMD ["node", "out"]
