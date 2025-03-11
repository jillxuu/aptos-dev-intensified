#!/bin/bash

# Script to run the topic-based chunking preprocessing

# Create necessary directories
mkdir -p data
mkdir -p logs

echo "Starting topic-based chunking preprocessing..."

# Run the preprocessing script
python scripts/preprocess_topic_chunks.py --docs-dir data/developer-docs/apps/nextra/pages/en --output data/enhanced_chunks.json

# Check if the preprocessing was successful
if [ $? -eq 0 ]; then
    echo "Preprocessing completed successfully!"
    echo "Enhanced chunks saved to data/enhanced_chunks.json"
    echo "You can now use the topic-based RAG provider in your application."
else
    echo "Preprocessing failed. Check the logs for details."
    exit 1
fi 