FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Calculate number of workers based on CPU cores
ENV WEB_CONCURRENCY=1

# Command to run the application with Gunicorn using the config file
CMD ["gunicorn", "main:app", "-c", "gunicorn.conf.py", "--log-config", "logging.conf"]