import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import os
import json
import logging

logger = logging.getLogger(__name__)


def initialize_firebase():
    """Initialize Firebase Admin SDK with credentials."""
    try:
        # Check if already initialized
        if not firebase_admin._apps:
            # First, try to get credentials from environment variable
            cred_json = os.getenv("FIREBASE_CREDENTIALS")
            if cred_json:
                try:
                    cred_dict = json.loads(cred_json)
                    cred = credentials.Certificate(cred_dict)
                except json.JSONDecodeError:
                    # If not JSON string, assume it's a path to the file
                    cred = credentials.Certificate(cred_json)
            else:
                # Fallback to looking for credentials file
                cred_path = os.getenv(
                    "GOOGLE_APPLICATION_CREDENTIALS", "firebase-credentials.json"
                )
                cred = credentials.Certificate(cred_path)

            # Initialize the app
            firebase_admin.initialize_app(cred)

        # Get Firestore client
        db = firestore.client()
        logger.info("Firebase initialized successfully")
        return db
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {str(e)}")
        raise


# Initialize Firestore client
try:
    db = initialize_firebase()
except Exception as e:
    logger.error(f"Could not initialize Firebase: {str(e)}")
    db = None
