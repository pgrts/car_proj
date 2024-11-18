# Use an official Python image as a base
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy only requirements initially to leverage Docker cache
COPY requirements.txt .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Environment variables for Flask and Postgres
ENV FLASK_APP=app.py
ENV FLASK_ENV=development
ENV DB_HOST=d
ENV DB_PORT=5432
ENV DB_USER=postgres
ENV DB_PASSWORD=${DB_PASSWORD} 

# Expose the port the app runs on
EXPOSE 5000

# Run Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
