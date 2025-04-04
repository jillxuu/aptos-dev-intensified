# Use Python 3.11 slim image
FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p ./data/generated/developer-docs && \
    mkdir -p ./data/generated/aptos-learn

# Clone documentation
RUN git clone --depth 1 https://github.com/aptos-labs/developer-docs.git data/developer-docs

# Set environment variables
ENV PORT=8080
ENV HOST=0.0.0.0

# Generate initial data
RUN python scripts/generate_data.py

# Run the application
CMD ["python", "-m", "app.main"] 