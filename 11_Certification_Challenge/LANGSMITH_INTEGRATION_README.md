# LangSmith Integration

This project includes complete integration with LangSmith to log all traces of questions launched from the frontend.

## Configuration

### Backend

LangSmith configuration is done in the backend as follows:

1. **Endpoint `/ask`** (`backend/main.py`):
   - Accepts an optional `langsmith_api_key` parameter
   - Configures LangSmith environment variables when the API key is provided

2. **Function `run_agentic_rag`** (`backend/agent_graph.py`):
   - Configures LangSmith at the start of processing
   - Passes the configuration through the entire processing chain

### Frontend

The frontend already includes an optional field for the LangSmith API key:

- Input field: "LangSmith API Key (optional)"
- Automatically sent to the backend when provided
- Not required for basic functionality

## Configured Environment Variables

When a LangSmith API key is provided, the following environment variables are configured:

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIE7-S11-Certification-Challenge-{uuid4().hex[0:8]}"
os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
```

## Operation

### With LangSmith
1. The user provides their LangSmith API key in the frontend
2. The backend configures LangSmith before processing the question
3. All LangChain operations are logged in LangSmith
4. The complete flow can be viewed in the LangSmith dashboard

### Without LangSmith
1. The user leaves the LangSmith API key field empty
2. The backend processes the question normally without trace logging
3. Basic functionality works the same

## Benefits

- **Complete traceability**: All LangChain operations are logged
- **Enhanced debugging**: You can see exactly what happened in each question
- **Optimization**: Bottlenecks can be identified and the flow optimized
- **Monitoring**: Performance and usage can be tracked

## Usage

1. Get your LangSmith API key from [LangSmith](https://smith.langchain.com/)
2. Start the backend: `cd backend && python main.py`
3. Start the frontend: `cd frontend && npm start`
4. Upload a PDF and ask a question
5. Provide your LangSmith API key (optional)
6. View the traces in your LangSmith dashboard

## Project Structure

```
├── backend/
│   ├── main.py              # Endpoint with LangSmith configuration
│   └── agent_graph.py       # Processing logic with LangSmith
├── frontend/
│   └── src/
│       └── App.js           # Interface with LangSmith API key field
└── test_langsmith_integration.py  # Test script
```

## Technical Notes

- LangSmith configuration is done at the start of processing to ensure all operations are logged
- Each session generates a unique project with a UUID to avoid conflicts
- The LangSmith API key field is completely optional
- The integration is transparent to the end user 