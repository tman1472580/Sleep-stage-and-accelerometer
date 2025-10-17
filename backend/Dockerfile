FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set port
ENV PORT=8080
EXPOSE 8080

# Use gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 1 --threads 2 --timeout 0 --access-logfile - --error-logfile - server:app
