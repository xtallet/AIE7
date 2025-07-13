from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import os
import sys
import io
import json
from typing import List

# Importar la lógica avanzada del notebook
from .main import get_compiled_super_graph
from langchain_core.messages import HumanMessage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - updated path for Docker container
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

class AskRequest(BaseModel):
    question: str
    api_key: str

class AskResponse(BaseModel):
    answer: str
    log: List[str]

async def stream_log_generator(question: str, api_key: str):
    """Generador para streaming del log en tiempo real."""
    try:
        # Use provided API key or fall back to environment variable
        openai_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            yield f"data: {json.dumps({'type': 'error', 'message': '❌ OpenAI API key is required. Please provide it in the frontend or set OPENAI_API_KEY environment variable.'})}\n\n"
            return
            
        os.environ["OPENAI_API_KEY"] = openai_key
        compiled_super_graph = get_compiled_super_graph()
        answer = None
        response_team_completed = False
        
        # Enviar mensaje inicial
        yield f"data: {json.dumps({'type': 'start', 'message': f'🚀 Starting processing of: {question}'})}\n\n"
        
        try:
            async for chunk in compiled_super_graph.astream(
                {"messages": [HumanMessage(content=question)], "next": ""},
                {"recursion_limit": 30},
            ):
                # Enviar información del chunk
                yield f"data: {json.dumps({'type': 'chunk', 'data': str(chunk)})}\n\n"
                
                for node, values in chunk.items():
                    # Enviar información del nodo
                    node_info = {
                        'type': 'node',
                        'node': node,
                        'values': str(values)
                    }
                    yield f"data: {json.dumps(node_info)}\n\n"
                    
                    # Log específico para Response team
                    if node == "Response team":
                        if not response_team_completed:
                            yield f"data: {json.dumps({'type': 'info', 'message': '🔄 Response team starting...'})}\n\n"
                        
                    # Capturar respuesta final del Response team
                    if node == "Response team" and "messages" in values:
                        last_msg = values["messages"][-1]
                        if hasattr(last_msg, 'content'):
                            content = last_msg.content
                            # Debug: Log the content being analyzed
                            yield f"data: {json.dumps({'type': 'debug', 'message': f'Analyzing Response team content: {content[:200]}...'})}\n\n"
                            
                            # Verificar si el contenido parece un post de LinkedIn completo
                            # Debe tener emojis Y hashtags Y formato de LinkedIn, NO ser solo investigación
                            has_emojis = any(emoji in content for emoji in ["🚀", "💡", "📊", "🌟", "✅", "🤖"])
                            has_hashtags = "#" in content
                            has_linkedin_format = any(keyword in content.lower() for keyword in ["linkedin", "post", "share", "thoughts", "game changer", "innovation", "let's discuss", "what are your thoughts"])
                            
                            # Mejorar la detección: si tiene emojis, hashtags y formato LinkedIn, es el post final
                            # incluso si menciona investigación (porque es normal en posts de LinkedIn sobre papers)
                            if has_emojis and has_hashtags and has_linkedin_format and "saved as" not in content.lower():
                                answer = content
                                response_team_completed = True
                                yield f"data: {json.dumps({'type': 'answer', 'content': answer})}\n\n"
                                yield f"data: {json.dumps({'type': 'info', 'message': '✅ Response team completed - LinkedIn post generated!'})}\n\n"
                                yield f"data: {json.dumps({'type': 'info', 'message': '🏁 Finalizing execution...'})}\n\n"
                            elif "saved as" in content.lower() or "outline" in content.lower():
                                # Este es un mensaje sobre archivos, no el post final
                                yield f"data: {json.dumps({'type': 'info', 'message': '📝 Response team working on content...'})}\n\n"
                            else:
                                # Contenido de investigación, no el post final
                                yield f"data: {json.dumps({'type': 'info', 'message': '🔍 Processing research content...'})}\n\n"
                    
                    # Detectar cuando el supervisor dice FINISH
                    if node == "supervisor" and "next" in values:
                        if values["next"] == "FINISH":
                            yield f"data: {json.dumps({'type': 'info', 'message': '✅ Supervisor detected completion - finishing execution'})}\n\n"
                            break
            
            # Enviar mensaje de finalización
            yield f"data: {json.dumps({'type': 'end', 'message': '🏁 Processing completed'})}\n\n"
            
        except Exception as e:
            error_msg = str(e)
            if "APIError" in error_msg or "server had an error" in error_msg.lower():
                yield f"data: {json.dumps({'type': 'error', 'message': '⚠️ Temporary OpenAI API error. Please try again in a few moments.'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': f'❌ Error during processing: {error_msg}'})}\n\n"
        
    except Exception as e:
        import traceback
        error_info = {
            'type': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }
        yield f"data: {json.dumps(error_info)}\n\n"

@app.get("/stream-ask")
async def stream_ask(question: str, api_key: str):
    """Endpoint de streaming para ver el progreso en tiempo real."""
    return StreamingResponse(
        stream_log_generator(question, api_key),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    # Redirigir stdout para capturar el log
    log_stream = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = log_stream
    try:
        # Use provided API key or fall back to environment variable
        openai_key = request.api_key or os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            return AskResponse(
                answer="❌ OpenAI API key is required. Please provide it in the frontend or set OPENAI_API_KEY environment variable.",
                log=["Error: No OpenAI API key provided"]
            )
            
        os.environ["OPENAI_API_KEY"] = openai_key
        # Ejecutar el meta-grafo completo
        compiled_super_graph = get_compiled_super_graph()
        answer = None
        log_lines = []
        response_team_completed = False
        
        print(f"🚀 Starting full processing of: {request.question}")
        print("=" * 50)
        
        async for chunk in compiled_super_graph.astream(
            {"messages": [HumanMessage(content=request.question)], "next": ""},
            {"recursion_limit": 30},
        ):
            print(f'📦 Chunk received: {chunk}')
            for node, values in chunk.items():
                print(f'🔄 Node: {node}')
                print(f'📊 Values: {values}')
                print(f"⏳ Processing node: '{node}'")
                
                if "messages" in values:
                    print("💬 Messages:")
                    for msg in values["messages"]:
                        if hasattr(msg, 'content'):
                            print(f"  - {msg.content[:100]}...")
                        else:
                            print(f"  - {str(msg)[:100]}...")
                else:
                    print(f"⚠️  No messages in: {values}")
                
                print("-" * 30)
                
                # Captura la última respuesta del modelo
                if node == "Response team" and "messages" in values:
                    last_msg = values["messages"][-1]
                    if hasattr(last_msg, 'content'):
                        content = last_msg.content
                        # Verificar si el contenido parece un post de LinkedIn completo
                        # Debe tener emojis Y hashtags Y formato de LinkedIn, NO ser solo investigación
                        has_emojis = any(emoji in content for emoji in ["🚀", "💡", "📊", "🌟", "✅", "🤖"])
                        has_hashtags = "#" in content
                        has_linkedin_format = any(keyword in content.lower() for keyword in ["linkedin", "post", "share", "thoughts", "game changer", "innovation", "let's discuss", "what are your thoughts"])
                        
                        # Mejorar la detección: si tiene emojis, hashtags y formato LinkedIn, es el post final
                        # incluso si menciona investigación (porque es normal en posts de LinkedIn sobre papers)
                        if has_emojis and has_hashtags and has_linkedin_format and "saved as" not in content.lower():
                            answer = content
                            response_team_completed = True
                            print(f"✅ Final answer captured: {answer[:100]}...")
                            print("✅ Response team completed - LinkedIn post generated!")
                        elif "saved as" in content.lower() or "outline" in content.lower():
                            print("📝 Response team working on content...")
                        else:
                            print("🔍 Processing research content...")
                
                # Detectar cuando el supervisor dice FINISH
                if node == "supervisor" and "next" in values:
                    if values["next"] == "FINISH":
                        print("✅ Supervisor detected completion - finishing execution")
                        break
        
        print("🏁 Processing completed")
        
        sys.stdout = sys_stdout
        log_stream.seek(0)
        log_lines = log_stream.read().splitlines()
        return AskResponse(answer=answer or "No answer generated", log=log_lines)
    except Exception as e:
        sys.stdout = sys_stdout
        import traceback
        error_details = f"Error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        return AskResponse(answer=error_details, log=log_stream.getvalue().splitlines()) 

@app.get("/")
async def read_root():
    """Serve the React app"""
    return FileResponse("backend/static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Multi-Agent LinkedIn Post Generator is running"} 