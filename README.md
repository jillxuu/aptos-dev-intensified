# Aptos RAG Assistant

A fine-tuned AI assistant powered by RAG (Retrieval-Augmented Generation) on Aptos data. This project combines modern LLMs (Claude/ChatGPT) with Aptos-specific knowledge to provide accurate and contextual responses.

## Features

- RAG implementation using LangChain
- Support for both Claude and ChatGPT models
- FastAPI backend for efficient API handling
- React frontend with modern UI
- Vector storage using ChromaDB
- Real-time chat interface

## Setup

### Backend
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
ANTHROPIC_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Frontend
1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

## Usage

1. Start the backend server:
```bash
uvicorn app.main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
```

3. Access the application at `http://localhost:5173`

## Project Structure

```
.
├── app/                    # Backend application
│   ├── main.py            # FastAPI application
│   ├── models.py          # Pydantic models
│   ├── rag/               # RAG implementation
│   └── routes/            # API routes
├── frontend/              # React frontend
├── data/                  # RAG data storage
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
``` 