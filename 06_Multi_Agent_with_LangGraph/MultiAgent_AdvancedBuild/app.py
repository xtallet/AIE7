import os
import sys
import uvicorn

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.api import app

if __name__ == "__main__":
    # Get port from environment variable (Hugging Face Spaces sets this)
    port = int(os.environ.get("PORT", 8000))
    
    # Run the FastAPI app
    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=port,
        reload=False
    ) 