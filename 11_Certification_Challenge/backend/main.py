from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import tempfile
import os
import getpass
from uuid import uuid4
from agent_graph import run_agentic_rag

app = FastAPI()

# Allow CORS for development (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskResponse(BaseModel):
    answer: str
    source: str  # "rag", "tavily", "arxiv"
    context: Optional[List[Dict[str, Any]]] = None
    raw_output: Optional[dict] = None

@app.post("/ask", response_model=AskResponse)
async def ask(
    pdf: UploadFile = File(...),
    question: str = Form(...),
    openai_api_key: str = Form(...),
    tavily_api_key: str = Form(...),
    langsmith_api_key: Optional[str] = Form(None)
):
    # Configure LangSmith if API key is provided
    if langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = f"AIE7-S11-Certification-Challenge-{uuid4().hex[0:8]}"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    
    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await pdf.read())
        tmp_path = tmp.name
    try:
        # Execute agentic RAG (vectorstore is created internally)
        result = await run_agentic_rag(question, tmp_path, openai_api_key, tavily_api_key, langsmith_api_key)
        return AskResponse(
            answer=result["answer"],
            source=result.get("source", "rag"),
            context=result.get("context"),
            raw_output=result.get("raw_output")
        )
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 