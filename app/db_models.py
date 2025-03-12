from datetime import datetime
from typing import List, Optional
from firebase_admin import firestore
from app.firebase_config import db
from app.models import ChatHistory, Feedback
import logging

logger = logging.getLogger(__name__)


class FirestoreChat:
    def __init__(self):
        self.db = db
        self.chats_ref = db.collection("chats")
        self.feedback_ref = db.collection("feedback")

    def _convert_to_dict(self, obj):
        """Convert Pydantic models to dictionary for Firestore."""
        if hasattr(obj, "dict"):
            data = obj.dict()
            # Ensure all nested objects are properly serialized
            for key, value in data.items():
                if isinstance(value, list):
                    # Handle lists of objects
                    data[key] = [
                        self._convert_to_dict(item) if hasattr(item, "dict") else item
                        for item in value
                    ]
                elif hasattr(value, "dict"):
                    # Handle nested objects
                    data[key] = self._convert_to_dict(value)
            return data
        return obj

    def _handle_timestamp(self, data: dict) -> dict:
        """Convert datetime objects to Firestore timestamps."""
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = firestore.SERVER_TIMESTAMP
        return data

    async def create_chat_history(self, history: ChatHistory) -> str:
        """Create a new chat history."""
        try:
            # Check if a chat with this ID already exists
            existing_chat = await self.get_chat_history(history.id)
            if existing_chat:
                logger.warning(
                    f"Chat history with ID {history.id} already exists, updating instead"
                )
                await self.update_chat_history(history)
                return history.id

            # Convert to dict and handle timestamps
            history_dict = self._convert_to_dict(history)
            history_dict = self._handle_timestamp(history_dict)

            # Add timestamp if not present
            if "timestamp" not in history_dict:
                history_dict["timestamp"] = firestore.SERVER_TIMESTAMP

            # Add to Firestore
            doc_ref = self.chats_ref.document(history.id)
            doc_ref.set(history_dict)

            return history.id
        except Exception as e:
            logger.error(f"Error creating chat history: {str(e)}")
            raise

    async def get_chat_history(self, chat_id: str) -> Optional[ChatHistory]:
        """Retrieve a chat history by ID."""
        try:
            doc = self.chats_ref.document(chat_id).get()
            if doc.exists:
                data = doc.to_dict()
                return ChatHistory(**data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            raise

    async def get_all_chat_histories(self) -> List[ChatHistory]:
        """Retrieve all chat histories."""
        try:
            docs = self.chats_ref.order_by(
                "timestamp", direction=firestore.Query.DESCENDING
            ).stream()
            return [ChatHistory(**doc.to_dict()) for doc in docs]
        except Exception as e:
            logger.error(f"Error retrieving all chat histories: {str(e)}")
            raise

    async def update_chat_history(self, history: ChatHistory):
        """Update an existing chat history."""
        try:
            logger.info(
                f"Updating chat history {history.id} with {len(history.messages)} messages"
            )
            history_dict = self._convert_to_dict(history)
            history_dict = self._handle_timestamp(history_dict)

            # Validate that all messages have content
            for i, message in enumerate(history_dict.get("messages", [])):
                if "content" not in message or message["content"] is None:
                    logger.warning(
                        f"Message {i} in chat {history.id} has no content, setting to empty string"
                    )
                    message["content"] = ""

            # Update the document
            self.chats_ref.document(history.id).set(history_dict, merge=True)
            logger.info(f"Successfully updated chat history {history.id}")
        except Exception as e:
            logger.error(f"Error updating chat history {history.id}: {str(e)}")
            raise

    async def delete_chat_history(self, chat_id: str):
        """Delete a chat history."""
        try:
            self.chats_ref.document(chat_id).delete()
        except Exception as e:
            logger.error(f"Error deleting chat history: {str(e)}")
            raise

    async def save_feedback(self, feedback: Feedback):
        """Save user feedback."""
        try:
            feedback_dict = self._convert_to_dict(feedback)
            feedback_dict = self._handle_timestamp(feedback_dict)

            # Add timestamp if not present
            if "timestamp" not in feedback_dict:
                feedback_dict["timestamp"] = firestore.SERVER_TIMESTAMP

            self.feedback_ref.add(feedback_dict)
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
            raise

    async def get_feedback_for_message(self, message_id: str) -> List[Feedback]:
        """Retrieve feedback for a specific message."""
        try:
            docs = self.feedback_ref.where("message_id", "==", message_id).stream()
            return [Feedback(**doc.to_dict()) for doc in docs]
        except Exception as e:
            logger.error(f"Error retrieving feedback: {str(e)}")
            raise

    async def get_client_chat_histories(self, client_id: str) -> List[ChatHistory]:
        """Retrieve all chat histories for a specific client."""
        try:
            # First try with just client_id filter without sorting
            docs = (
                self.chats_ref.where("client_id", "==", client_id)
                .limit(100)  # Add reasonable limit
                .stream()
            )

            histories = []
            for doc in docs:
                try:
                    history_dict = doc.to_dict()
                    histories.append(ChatHistory(**history_dict))
                except Exception as e:
                    logger.error(f"Error parsing chat history document: {e}")
                    continue

            # Sort in memory if we have results
            if histories:
                histories.sort(key=lambda x: x.timestamp, reverse=True)

            return histories

        except Exception as e:
            logger.error(f"Error retrieving client chat histories: {str(e)}")
            return []  # Return empty list instead of raising


# Create a global instance
firestore_chat = FirestoreChat()
