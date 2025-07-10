import React, { useState } from "react";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [toolsUsed, setToolsUsed] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAsk = async () => {
    setLoading(true);
    setAnswer("");
    setToolsUsed([]);
    setError("");
    try {
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
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
    <div style={{ maxWidth: 600, margin: "auto", padding: 32 }}>
      <h1>Science Research Agent</h1>
      <input
        value={question}
        onChange={e => setQuestion(e.target.value)}
        placeholder="Type your question..."
        style={{ width: "80%", padding: 8, fontSize: 16 }}
        onKeyDown={e => { if (e.key === "Enter") handleAsk(); }}
      />
      <button onClick={handleAsk} disabled={loading || !question} style={{ marginLeft: 8 }}>
        {loading ? "Searching..." : "Ask"}
      </button>
      {error && <div style={{ color: "red", marginTop: 16 }}>{error}</div>}
      <div style={{ marginTop: 32 }}>
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
  );
}

export default App;
