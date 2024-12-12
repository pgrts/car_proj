FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install system dependencies and Supervisor
RUN apt-get update && apt-get install -y nginx supervisor && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* \
    rm /etc/nginx/sites-enabled/default
    
# Copy Nginx and Supervisor configurations to their locations
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports for Nginx and Gunicorn
EXPOSE 80 8080

# Start Supervisor to manage both Nginx and Gunicorn
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
