# Agentic RAG - PDF Question Answering System

A complete web application that allows uploading PDFs, asking questions and getting answers using an End-to-End Agentic RAG system with LangGraph, which combines RAG (Retrieval-Augmented Generation), web searches with Tavily and academic searches with Arxiv.

## 🚀 Features

- **FastAPI Backend**: PDF processing, vectorization and agentic RAG system
- **React Frontend**: Modern interface for uploading PDFs and asking questions
- **Agentic RAG**: Intelligent system that decides when to use RAG, Tavily or Arxiv
- **Context visualization**: Shows PDF fragments used to answer
- **Traceability**: Indicates the source of the response (RAG, Tavily, Arxiv)

## 📁 Project Structure

```
├── backend/
│   ├── main.py              # FastAPI API
│   └── agent_graph.py       # Agentic RAG logic with LangGraph
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   └── App.css          # Modern styles
│   └── package.json
├── pyproject.toml           # Python dependencies
└── README.md
```

## 🛠️ Installation and Configuration

### Prerequisites

- Python 3.13+
- Node.js 18+
- API Keys:
  - OpenAI API Key
  - Tavily API Key
  - LangSmith API Key (optional)

### Backend (FastAPI)

1. **Install Python dependencies**:
   ```bash
   uv sync
   ```

2. **Run the backend**:
   ```bash
   cd backend
   python main.py
   ```
   
   The server will be available at: `http://localhost:8000`

3. **Automatic documentation**:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

### Frontend (React)

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Run the frontend**:
   ```bash
   npm start
   ```
   
   The application will be available at: `http://localhost:3000`

## 🎯 Usage

1. **Open the application**: Navigate to `http://localhost:3000`

2. **Upload a PDF**: Select a PDF file from your computer

3. **Write a question**: Enter your question about the PDF content

4. **Configure API Keys**:
   - **OpenAI API Key**: Required for language processing
   - **Tavily API Key**: Required for web searches
   - **LangSmith API Key**: Optional for logging and debugging

5. **Send**: Click "Send Question" and wait for the response

6. **View results**: The application will show:
   - The model's response
   - The source used (RAG, Tavily, Arxiv)
   - The PDF fragments used as context

## 🔧 API Endpoints

### POST /ask

**Parameters** (multipart/form-data):
- `pdf`: PDF file
- `question`: User's question
- `openai_api_key`: OpenAI API Key
- `tavily_api_key`: Tavily API Key
- `langsmith_api_key`: LangSmith API Key (optional)

**Response**:
```json
{
  "answer": "Model response...",
  "source": "rag|tool",
  "context": [
    {
      "page": 14,
      "snippet": "PDF fragment used...",
      "source": "filename.pdf"
    }
  ],
  "raw_output": {...}
}
```

## 🧠 Agentic RAG System

The system uses LangGraph to implement an agentic flow that:

1. **pre_rag_context**: Retrieves relevant context from the PDF
2. **agent**: Processes the question with context
3. **action**: If necessary, uses external tools (Tavily, Arxiv)
4. **should_continue**: Decides whether to continue with tools or finish

### Available Tools

- **RAG Tool**: Search in the vectorized PDF
- **Tavily**: Real-time web searches
- **Arxiv**: Academic paper searches

## 🎨 Frontend Features

- **Modern design**: Interface with gradients and glassmorphism effects
- **Responsive**: Works on mobile and desktop devices
- **Validation**: Verifies PDF files and required fields
- **Visual feedback**: Loading states and error handling
- **Context visualization**: Shows the fragments used

## 🔍 Usage Example

1. Upload an insurance policy PDF
2. Question: "What is the coverage for radiation damage?"
3. The system:
   - Searches the PDF using RAG
   - If it doesn't find enough information, searches the web with Tavily
   - Returns the response with the context used

## 🛡️ Security

- API keys are sent securely to the backend
- They are not stored in the frontend
- The backend processes PDFs temporarily and deletes them

## 🚀 Deployment

### Backend (Production)
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Frontend (Production)
```bash
cd frontend
npm run build
# Serve static files
```

## 📝 Technical Notes

- **Vectorstore**: Qdrant in memory for each session
- **Embeddings**: OpenAI text-embedding-3-small
- **Model**: GPT-4.1-nano for responses
- **Chunking**: 1000 characters with 200 overlap
- **CORS**: Configured for local development

## 🤝 Contributing

1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is under the MIT License. 