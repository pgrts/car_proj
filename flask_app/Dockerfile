FROM python:3.9-slim

# Set the working directory to /app inside the container
WORKDIR /app

# Copy the current directory contents into the container (excluding the ones in .dockerignore)
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the application will run on
EXPOSE 8000

# Command to run the Flask app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]