import random

# Test responses for variety
TEST_RESPONSES = [
    "Here's a test response about {topic}! I'm in test mode so I'll keep it simple.",
    "Let me tell you about {topic} (but not really, I'm in test mode).",
    "Interesting question about {topic}! In test mode, I'll just acknowledge that.",
    "I would love to explain {topic}, but I'm in test mode right now.",
    "That's a great question about {topic}! (Test mode active)",
]


def generate_test_response(message: str, chat_id: str = None) -> str:
    """Generate a random test response with the chat ID and topic."""
    # Extract a topic from the message (first few words)
    topic = " ".join(message.split()[:3]) + "..."

    # Pick a random response template
    template = random.choice(TEST_RESPONSES)
    response = template.format(topic=topic)

    # Add chat ID info if provided
    if chat_id:
        response += f"\n\n(Continuing Chat ID: {chat_id})"

    return response
