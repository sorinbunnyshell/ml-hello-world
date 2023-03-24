FROM nginx:alpine

ARG API_URL

COPY nginx.conf /etc/nginx/nginx.conf
COPY . /usr/share/nginx/html

RUN sed -i "s#<<API_URL>>#${API_URL}#g" /usr/share/nginx/html/js/app.js
