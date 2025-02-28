#!/bin/bash

# Exit on any error
set -e

# Configuration
PROJECT_ID="aptos-jill-playground"  # Replace with your GCP project ID
REGION="us-central1"          # Replace with your preferred region
SERVICE_NAME="aptos-dev-assistant"
REPOSITORY="cloud-run-source-deploy"
IMAGE_NAME="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$SERVICE_NAME"
SECRET_NAME="firebase-credentials"

# Check required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable is required"
    exit 1
fi

# Configure Docker to use Google Cloud credentials
echo "🔑 Configuring Docker authentication..."
gcloud auth configure-docker $REGION-docker.pkg.dev --quiet

# Check if FIREBASE_CREDENTIALS file exists
if [ ! -f "$FIREBASE_CREDENTIALS" ]; then
    echo "❌ Error: FIREBASE_CREDENTIALS environment variable must point to a valid JSON file"
    exit 1
fi

# Check if secret already exists
if ! gcloud secrets describe $SECRET_NAME --project $PROJECT_ID &>/dev/null; then
    echo "🔐 Creating new secret for Firebase credentials..."
    gcloud secrets create $SECRET_NAME \
        --project $PROJECT_ID \
        --replication-policy="automatic" \
        --data-file="$FIREBASE_CREDENTIALS"
else
    echo "🔄 Using existing Firebase credentials secret..."
fi

# Build the Docker image
echo "🏗️ Building Docker image..."
docker build --platform linux/amd64 -t $IMAGE_NAME .

# Push the image to Artifact Registry
echo "⬆️ Pushing image to Artifact Registry..."
docker push $IMAGE_NAME

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --project $PROJECT_ID \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10 \
  --set-env-vars="OPENAI_API_KEY=${OPENAI_API_KEY}" \
  --set-env-vars="CHAT_TEST_MODE=false" \
  --set-secrets="FIREBASE_CREDENTIALS=$SECRET_NAME:latest"

echo "✅ Deployment complete!"
echo "🌐 Service URL: $(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')" 