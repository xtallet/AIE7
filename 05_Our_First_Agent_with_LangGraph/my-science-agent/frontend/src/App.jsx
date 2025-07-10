import React, { useState } from "react";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [toolsUsed, setToolsUsed] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [openaiKey, setOpenaiKey] = useState("");
  const [tavilyKey, setTavilyKey] = useState("");

  // Detect if we're in development (localhost) or production (Hugging Face Spaces)
  const isDevelopment = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  const apiUrl = isDevelopment ? 'http://localhost:8000/ask' : '/ask';

  const handleAsk = async () => {
    setLoading(true);
    setAnswer("");
    setToolsUsed([]);
    setError("");
    try {
      const res = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, openai_api_key: openaiKey, tavily_api_key: tavilyKey }),
      });
      if (!res.ok) throw new Error("Failed to get response from backend");
      const data = await res.json();
      setAnswer(data.response);
      setToolsUsed(data.tools_used || []);
    } catch (err) {
      setError(err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 1200, margin: "auto", padding: 32 }}>
      <h1>Agentic RAG Powered by LangChain</h1>
      <div style={{ display: "flex", gap: "40px", alignItems: "flex-start" }}>
        <div style={{ flex: "1", maxWidth: "400px" }}>
          <input
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder="Type your question..."
            style={{ width: "100%", padding: 8, fontSize: 16, marginBottom: 8 }}
            onKeyDown={e => { if (e.key === "Enter") handleAsk(); }}
          />
          <input
            value={openaiKey}
            onChange={e => setOpenaiKey(e.target.value)}
            placeholder="OpenAI API Key"
            style={{ width: "100%", padding: 8, fontSize: 16, marginBottom: 8 }}
            type="password"
            autoComplete="off"
          />
          <input
            value={tavilyKey}
            onChange={e => setTavilyKey(e.target.value)}
            placeholder="Tavily API Key"
            style={{ width: "100%", padding: 8, fontSize: 16, marginBottom: 8 }}
            type="password"
            autoComplete="off"
          />
          <button onClick={handleAsk} disabled={loading || !question} style={{ marginLeft: 8 }}>
            {loading ? "Searching..." : "Ask"}
          </button>
          {error && <div style={{ color: "red", marginTop: 16 }}>{error}</div>}
        </div>
        <div style={{ flex: "1", background: "rgba(255,255,255,0.9)", padding: "20px", borderRadius: "8px", minHeight: "200px", color: "#333" }}>
          <strong>Answer:</strong>
          <p>{answer}</p>
          {toolsUsed.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <strong>Tools used:</strong>
              <ul>
                {toolsUsed.map(tool => (
                  <li key={tool}>{tool}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
