# Use a secure, minimal base image
FROM python:3.11-slim

# Set environment variables
# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app

# Set working directory
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run training to ensure model artifact exists in the container
# In a real CI/CD, this might be a separate build step or downloaded from a registry
RUN python train.py

# Expose the application port
EXPOSE 8000

# Run the application with Uvicorn configured for multi-worker concurrency
# Using 4 workers as requested
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
