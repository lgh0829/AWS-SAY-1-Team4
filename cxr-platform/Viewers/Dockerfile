# syntax=docker/dockerfile:1.7-labs
# This dockerfile is used to publish the `ohif/app` image on dockerhub.
#
# It's a good example of how to build our static application and package it
# with a web server capable of hosting it as static content.
#
# docker build
# --------------
# If you would like to use this dockerfile to build and tag an image, make sure
# you set the context to the project's root directory:
# https://docs.docker.com/engine/reference/commandline/build/
#
#
# SUMMARY
# --------------
# This dockerfile has two stages:
#
# 1. Building the React application for production
# 2. Setting up our Nginx (Alpine Linux) image w/ step one's output
#


# syntax=docker/dockerfile:1.7-labs
# This dockerfile is used to publish the `ohif/app` image on dockerhub.
#
# It's a good example of how to build our static application and package it
# with a web server capable of hosting it as static content.
#
# docker build
# --------------
# If you would like to use this dockerfile to build and tag an image, make sure
# you set the context to the project's root directory:
# https://docs.docker.com/engine/reference/commandline/build/
#
#
# SUMMARY
# --------------
# This dockerfile is used as an input for a second stage to make things run faster.
#


# Stage 1: Build the application
# docker build -t ohif/viewer:latest .
# Copy Files
FROM node:20.18.1-slim as builder

RUN apt-get update && apt-get install -y build-essential python3


RUN mkdir /usr/src/app
WORKDIR /usr/src/app
RUN npm install -g bun
# RUN npm install -g lerna@7.4.2
ENV PATH=/usr/src/app/node_modules/.bin:$PATH

# Do an initial install and then a final install
COPY package.json yarn.lock preinstall.js lerna.json ./
COPY --parents ./addOns/package.json ./addOns/*/*/package.json ./extensions/*/package.json ./modes/*/package.json ./platform/*/package.json ./
# Run the install before copying the rest of the files

RUN bun pm cache rm
RUN bun install
# Copy the local directory
COPY --link --exclude=yarn.lock --exclude=package.json --exclude=Dockerfile . .


# Build here
# After install it should hopefully be stable until the local directory changes
ENV QUICK_BUILD true
# ENV GENERATE_SOURCEMAP=false
# ARG APP_CONFIG=./platform/app/public/config/default.js
ENV APP_CONFIG=${APP_CONFIG}
ARG PUBLIC_URL=/
ENV PUBLIC_URL=${PUBLIC_URL}
RUN bun run show:config
RUN bun run build

# Precompress files
RUN chmod u+x .docker/compressDist.sh
RUN ./.docker/compressDist.sh


# Stage 3: Bundle the built application into a Docker container
FROM nginxinc/nginx-unprivileged:1.27-alpine as final
ARG PUBLIC_URL=/
ENV PUBLIC_URL=${PUBLIC_URL}
ARG PORT=80
ENV PORT=${PORT}
RUN rm /etc/nginx/conf.d/default.conf

# 권한 문제 해결을 위해 루트 사용자로 시작
USER root
COPY .docker/Viewer-v3.x /usr/src
RUN chmod 777 /usr/src/entrypoint.sh

# 앱 설정 파일 디렉토리 생성
RUN mkdir -p /usr/share/nginx/html/config
# 기본 설정 파일 복사
COPY --from=builder /usr/src/app/platform/app/public/config/docker-nginx-orthanc.js /usr/share/nginx/html/config/default.js
# app-config.js 파일은 복사만 하고 볼륨 마운트로 대체할 수 있도록 함
COPY --from=builder /usr/src/app/platform/app/public/config/default.js /usr/share/nginx/html/app-config.js

# 앱 파일 복사
COPY --from=builder /usr/src/app/platform/app/dist /usr/share/nginx/html${PUBLIC_URL}
COPY --from=builder /usr/src/app/platform/app/dist/dicom-microscopy-viewer /usr/share/nginx/html/dicom-microscopy-viewer

# 디렉토리 생성 및 권한 설정 - 모든 nginx 디렉토리에 쓰기 권한 부여
RUN mkdir -p /var/run \
 && chown -R nginx:nginx /var/run \
 && chown -R nginx:nginx /usr/share/nginx/html \
 && chown -R nginx:nginx /usr/src \
 && chmod -R 755 /usr/share/nginx/html

# 다시 nginx 사용자로 전환
USER nginx
ENTRYPOINT ["/usr/src/entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
