from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chat
from app.models import initialize_models, load_aptos_docs, load_documents
import os

app = FastAPI(
    title="Aptos Dev Assistant API",
    description="API for the Aptos Developer Assistant chatbot",
    version="1.0.0"
)

# Configure CORS
origins = [
    "http://localhost:5173",  # Local development
    "http://localhost:3000",  # Local production build
    "https://aptos-dev-intensified.vercel.app",  # Production frontend
    "https://aptos-dev-assistant-sdlgryoz4q-uc.a.run.app",  # Production backend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    # Initialize models and load documents
    initialize_models()
    load_aptos_docs()
    load_documents()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True) 