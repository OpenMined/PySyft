FROM node:16 as build-stage
WORKDIR /app
COPY package.json .
ENV ENVIRONMENT=${FRONTEND_ENV}
ENV NEXT_TELEMETRY_DISABLED=1
RUN yarn install
COPY . .

FROM build-stage as grid-ui-development
CMD ["/usr/local/bin/node", "--max_old_space_size=4096", "/app/node_modules/.bin/next", "dev", "-p", "80"]

FROM build-stage as grid-ui-production
ENV PORT=80
WORKDIR /app
RUN yarn build
RUN yarn export

FROM nginx:1.15
COPY --from=grid-ui-production /app/out /usr/share/nginx/html
COPY ./docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY ./docker/nginx-backend-not-found.conf /etc/nginx/extra-conf.d/backend-not-found

