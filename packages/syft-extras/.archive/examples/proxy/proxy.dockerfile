# docker build -f proxy/proxy.dockerfile -t syftbox-proxy proxy
# docker run -d -p 80:80 -p 443:443 -v $(pwd)/proxy/server_nginx.conf:/etc/nginx/nginx.conf --name server-syftbox-proxy syftbox-proxy

# client
# docker run -d -p 9980:80 -p 9943:443 -v $(pwd)/proxy/client_nginx.conf:/etc/nginx/nginx.conf --name client-syftbox-proxy syftbox-proxy


# bore + tls
# ssh -i "./keys/test-madhava-dns_key.pem" "azureuser@20.38.32.165"
# bore local 9080 --to 20.38.32.165 -p 6000
# bore local 9443 --to 20.38.32.165 -p 6001


# openssl genrsa -out syftbox.localhost.key 2048
# openssl req -new -key syftbox.localhost.key -out syftbox.localhost.csr
# openssl x509 -req -in syftbox.localhost.csr -signkey syftbox.localhost.key -out syftbox.localhost.crt -days 365 -extfile <(printf "subjectAltName=DNS:syftbox.localhost,DNS:*.syftbox.localhost")

# Use an official Nginx base image
FROM nginx:alpine

# Install inotify-tools to monitor file changes
RUN apk update && apk add inotify-tools

# Copy your custom Nginx configuration (optional, if needed)
# COPY ./nginx.conf /etc/nginx/nginx.conf

COPY ./certs/* /etc/nginx/

# Copy the hot-reload bash script to the container
COPY ./start.sh /usr/local/bin/nginx-reload.sh

# Make the bash script executable
RUN chmod +x /usr/local/bin/nginx-reload.sh

# Expose port 443 for HTTPS
EXPOSE 443

# Expose port 80 for HTTP traffic
EXPOSE 80

# Start the bash script in the background and Nginx in the foreground
CMD ["/bin/sh", "-c", "/usr/local/bin/nginx-reload.sh & nginx -g 'daemon off;'"]
