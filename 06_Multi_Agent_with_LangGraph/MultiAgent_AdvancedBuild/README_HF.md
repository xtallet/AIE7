# Multi-Agent LinkedIn Post Generator

This Space demonstrates a sophisticated multi-agent system that generates professional LinkedIn posts based on machine learning research papers.

## How to Use

1. **Enter your OpenAI API key** in the input field
2. **Ask a question** about a machine learning research topic
3. **Watch the real-time generation** process
4. **Get your professional LinkedIn post** with emojis and hashtags

## Example Questions

- "Write a LinkedIn post about transformer models for low-resource settings"
- "Create a post about recent advances in machine learning interpretability"
- "Generate a LinkedIn post about efficient AI models"

## Architecture

The system uses LangGraph to orchestrate multiple AI agents:

- **Research Team**: Searches arXiv and analyzes research papers
- **Response Team**: Creates engaging LinkedIn posts with proper formatting
- **Supervisor**: Coordinates the workflow between teams

## Features

- ✅ Real-time streaming of the generation process
- ✅ Professional LinkedIn formatting with emojis and hashtags
- ✅ Research-based content generation
- ✅ Modern React UI
- ✅ Multi-agent orchestration

## Technical Stack

- **Backend**: FastAPI, LangChain, LangGraph, OpenAI
- **Frontend**: React, Vite, Server-Sent Events
- **Deployment**: Docker on Hugging Face Spaces

## API Endpoints

- `GET /`: Main application interface
- `GET /health`: Health check
- `POST /ask`: Generate LinkedIn post (non-streaming)
- `GET /stream-ask`: Generate LinkedIn post (streaming)

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (set as a secret in Space settings)

## Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
python -m uvicorn api:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

## License

Apache 2.0 