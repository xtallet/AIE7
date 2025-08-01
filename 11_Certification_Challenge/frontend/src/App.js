import React, { useState } from 'react';
import './App.css';

function App() {
  const [pdfFile, setPdfFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [openaiApiKey, setOpenaiApiKey] = useState('');
  const [tavilyApiKey, setTavilyApiKey] = useState('');
  const [langsmithApiKey, setLangsmithApiKey] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Function to detect which tools were used based on response content
  const detectSourcesUsed = (answer) => {
    const answerLower = answer.toLowerCase();
    const sources = [];
    
    // Check for RAG usage (mentions of document content, policy details, etc.)
    if (answerLower.includes('policy') || 
        answerLower.includes('document') || 
        answerLower.includes('clause') ||
        answerLower.includes('umr') ||
        answerLower.includes('coverage') ||
        answerLower.includes('exclusion') ||
        answerLower.includes('institute') ||
        answerLower.includes('endorsement') ||
        answerLower.includes('2015') ||
        answerLower.includes('provided context') ||
        answerLower.includes('based on the provided')) {
      sources.push('RAG');
    }
    
    // Check for external search usage (mentions of current information, standards, etc.)
    if (answerLower.includes('search results') || 
        answerLower.includes('according to recent') ||
        answerLower.includes('current market') ||
        answerLower.includes('latest developments') ||
        answerLower.includes('recent studies') ||
        answerLower.includes('market analysis') ||
        answerLower.includes('industry standards') ||
        answerLower.includes('as of 2024') ||
        answerLower.includes('as of 2025') ||
        answerLower.includes('current standards') ||
        answerLower.includes('modern policies') ||
        answerLower.includes('evolved significantly') ||
        answerLower.includes('by 2024') ||
        answerLower.includes('in 2024') ||
        answerLower.includes('naic reports') ||
        answerLower.includes('industry guidelines') ||
        answerLower.includes('compared to 2015') ||
        answerLower.includes('since 2015') ||
        answerLower.includes('over time') ||
        answerLower.includes('have evolved') ||
        answerLower.includes('current cybersecurity') ||
        answerLower.includes('modern policies')) {
      sources.push('Tavily');
    }
    
    // If no sources detected, assume RAG was used
    if (sources.length === 0) {
      sources.push('RAG');
    }
    
    return sources.join(' + ');
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') {
      setPdfFile(file);
      setError('');
    } else {
      setError('Please select a valid PDF file.');
      setPdfFile(null);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    
    if (!pdfFile || !question || !openaiApiKey || !tavilyApiKey) {
      setError('Please complete all required fields.');
      return;
    }

    setLoading(true);
    setError('');
    setResponse(null);

    const formData = new FormData();
    formData.append('pdf', pdfFile);
    formData.append('question', question);
    formData.append('openai_api_key', openaiApiKey);
    formData.append('tavily_api_key', tavilyApiKey);
    if (langsmithApiKey) {
      formData.append('langsmith_api_key', langsmithApiKey);
    }

    try {
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setResponse(data);
    } catch (err) {
      setError(`Error processing request: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>End-to-End Agentic RAG for Insurance Documents</h1>
        <p>Upload a PDF, ask a question and get answers using RAG, Tavily and Arxiv</p>
      </header>

      <main className="App-main">
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="form-group">
            <label htmlFor="pdf">Select PDF:</label>
            <input
              type="file"
              id="pdf"
              accept=".pdf"
              onChange={handleFileChange}
              required
            />
            {pdfFile && <p className="file-info">Selected file: {pdfFile.name}</p>}
          </div>

          <div className="form-group">
            <label htmlFor="question">Question:</label>
            <textarea
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Write your question here..."
              required
              rows="3"
            />
          </div>

          <div className="form-group">
            <label htmlFor="openai-api-key">OpenAI API Key:</label>
            <input
              type="password"
              id="openai-api-key"
              value={openaiApiKey}
              onChange={(e) => setOpenaiApiKey(e.target.value)}
              placeholder="sk-..."
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="tavily-api-key">Tavily API Key:</label>
            <input
              type="password"
              id="tavily-api-key"
              value={tavilyApiKey}
              onChange={(e) => setTavilyApiKey(e.target.value)}
              placeholder="tvly-..."
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="langsmith-api-key">LangSmith API Key (optional):</label>
            <input
              type="password"
              id="langsmith-api-key"
              value={langsmithApiKey}
              onChange={(e) => setLangsmithApiKey(e.target.value)}
              placeholder="ls_..."
            />
          </div>

          <button type="submit" disabled={loading} className="submit-btn">
            {loading ? 'Processing...' : 'Send Question'}
          </button>
        </form>

        {error && (
          <div className="error-message">
            <p>{error}</p>
          </div>
        )}

        {response && (
          <div className="response-container">
            <h2>Response</h2>
            <div className="response-info">
              <p><strong>Source:</strong> {detectSourcesUsed(response.answer)}</p>
              <p><strong>Answer:</strong></p>
              <div className="answer-text">{response.answer}</div>
            </div>

            {response.context && response.context.length > 0 && (
              <div className="context-section">
                <h3>Context Used:</h3>
                {response.context.map((ctx, index) => (
                  <div key={index} className="context-item">
                    <p><strong>Page:</strong> {ctx.page || 'N/A'}</p>
                    <p><strong>Source:</strong> {ctx.source}</p>
                    <p><strong>Excerpt:</strong></p>
                    <div className="snippet">{ctx.snippet}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
