"""
API routes package.
"""

from fastapi import APIRouter
from app.routes.chat import router

# Export the chat router as the default router
__all__ = ["router"]
