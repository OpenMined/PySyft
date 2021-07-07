# Stage 0, "build-stage", based on Node.js, to build and compile the frontend
FROM node:16 as build-stage

WORKDIR /app

COPY package.json /app/

RUN npm install

COPY ./ /app/

ARG ENVIRONMENT=production

ENV APP_ENV=${FRONTEND_ENV}

RUN npm run build
RUN npm run export

# Stage 1, based on Nginx, to have only the compiled app, ready for production with Nginx
FROM nginx:1.15
COPY --from=build-stage /app/out/ /usr/share/nginx/html
COPY ./docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY ./docker/nginx-backend-not-found.conf /etc/nginx/extra-conf.d/backend-not-found.conf
