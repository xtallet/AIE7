---
title: Multi-Agent LinkedIn Post Generator
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
app_port: 7860
---

# Multi-Agent LinkedIn Post Generator

A sophisticated multi-agent system that generates professional LinkedIn posts based on machine learning research papers. The system uses a research team to gather information and a response team to create engaging LinkedIn content.

## Features

- **Research Team**: Searches and analyzes machine learning research papers
- **Response Team**: Creates professional LinkedIn posts with emojis and hashtags
- **Real-time Streaming**: See the generation process in real-time
- **Modern UI**: React frontend with real-time updates

## Architecture

The system uses LangGraph to orchestrate multiple AI agents:

1. **Research Team**: Searches arXiv for relevant papers and provides summaries
2. **Response Team**: Creates engaging LinkedIn posts with proper formatting
3. **Supervisor**: Coordinates between teams and ensures proper workflow

## Local Development

### Backend
```bash
cd MultiAgent_AdvancedBuild/backend
pip install -r requirements.txt
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd MultiAgent_AdvancedBuild/frontend
npm install
npm run dev
```

## Deployment to Hugging Face Spaces

This application is configured for deployment on Hugging Face Spaces using Docker.

### Requirements

- Docker
- Hugging Face account

### Deployment Steps

1. **Create a new Space** on Hugging Face:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Docker" as the SDK
   - Set the license to "Apache 2.0"

2. **Upload the code**:
   - Clone your space repository
   - Copy all files from `MultiAgent_AdvancedBuild/` to the space repository
   - Commit and push the changes

3. **Environment Variables**:
   - Add your OpenAI API key as a secret in the Space settings
   - The application will use the `OPENAI_API_KEY` environment variable

### Docker Configuration

The `Dockerfile` is configured to:
- Build the React frontend
- Serve static files through FastAPI
- Run the application on port 8000

### API Endpoints

- `GET /`: Serves the React frontend
- `GET /health`: Health check endpoint
- `POST /ask`: Generate LinkedIn post (non-streaming)
- `GET /stream-ask`: Generate LinkedIn post (streaming)

## Usage

1. Enter your OpenAI API key
2. Provide a question about a machine learning research topic
3. The system will research the topic and generate a professional LinkedIn post
4. View the real-time generation process and final result

## Technologies Used

- **Backend**: FastAPI, LangChain, LangGraph, OpenAI
- **Frontend**: React, Vite, Server-Sent Events
- **Deployment**: Docker, Hugging Face Spaces 