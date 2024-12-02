FROM python:3.9-slim

# Copy the app and data folder into the container
COPY . /app

# Make sure the data directory is included
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create the directory for the SQLite database
RUN mkdir -p /app/data

# Expose the port that Gunicorn will run on
EXPOSE 8000

# Run Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]