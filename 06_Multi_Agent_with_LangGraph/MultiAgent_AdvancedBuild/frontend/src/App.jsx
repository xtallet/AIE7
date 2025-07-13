import { useState } from 'react';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [answer, setAnswer] = useState('');
  const [log, setLog] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setAnswer('');
    setLog([]);
    
    try {
      // Use streaming to see real-time progress
      const eventSource = new EventSource(
        `http://localhost:8000/stream-ask?question=${encodeURIComponent(question)}&api_key=${encodeURIComponent(apiKey)}`
      );
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          switch (data.type) {
            case 'start':
              setLog(prev => [...prev, data.message]);
              break;
            case 'chunk':
              setLog(prev => [...prev, `📦 Chunk: ${data.data}`]);
              break;
            case 'node':
              setLog(prev => [...prev, `🔄 Node ${data.node}: ${data.values}`]);
              break;
            case 'info':
              setLog(prev => [...prev, data.message]);
              break;
            case 'answer':
              setAnswer(data.content);
              setLog(prev => [...prev, `✅ Final response: ${data.content.substring(0, 100)}...`]);
              break;
            case 'end':
              setLog(prev => [...prev, data.message]);
              eventSource.close();
              setLoading(false);
              break;
            case 'error':
              setError(`${data.message}\n${data.traceback}`);
              eventSource.close();
              setLoading(false);
              break;
            default:
              setLog(prev => [...prev, `📝 ${JSON.stringify(data)}`]);
          }
        } catch (err) {
          setLog(prev => [...prev, `❌ Error parsing: ${err.message}`]);
        }
      };
      
      eventSource.onerror = (error) => {
        setError('Streaming connection error');
        eventSource.close();
        setLoading(false);
      };
      
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Multi-Agent System Demo</h1>
      <form onSubmit={handleSubmit} className="form">
        <label>
          Question:
          <textarea
            value={question}
            onChange={e => setQuestion(e.target.value)}
            rows={3}
            required
            placeholder="Enter your question for the agents..."
          />
        </label>
        <label>
          OpenAI API Key:
          <input
            type="password"
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            required
            placeholder="sk-..."
          />
        </label>
        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Send'}
        </button>
      </form>
      
      {error && <div className="error">{error}</div>}
      
      {loading && (
        <div className="loading">
          <h3>Processing with multi-agents...</h3>
          <p>The system is executing the complete research and writing workflow.</p>
          <p>This may take several minutes. Please be patient.</p>
          <div className="spinner">⏳</div>
        </div>
      )}
      
      {answer && (
        <div className="result">
          <h2>Final Response</h2>
          <pre>{answer}</pre>
        </div>
      )}
      
      {log.length > 0 && (
        <div className="log">
          <h2>Process Log (real-time)</h2>
          <pre style={{ maxHeight: 400, overflow: 'auto', background: '#222', color: '#eee', padding: 10 }}>
            {log.join('\n')}
          </pre>
        </div>
      )}
    </div>
  );
}

export default App;
