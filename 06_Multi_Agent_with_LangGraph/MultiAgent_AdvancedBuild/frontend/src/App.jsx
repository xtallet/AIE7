import React, { useState } from 'react';
import QuestionForm from './components/QuestionForm';
import LogViewer from './components/LogViewer';
import AnswerBox from './components/AnswerBox';
import './App.css';

function App() {
  const [log, setLog] = useState([]);
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAsk = async ({ question, apiKey }) => {
    setLoading(true);
    setLog([]);
    setAnswer('');
    setError('');
    try {
      const response = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, api_key: apiKey })
      });
      if (!response.ok) {
        throw new Error('Error en la consulta: ' + response.statusText);
      }
      const data = await response.json();
      setAnswer(data.answer || '');
      setLog(data.log || []);
      setLoading(false);
    } catch (err) {
      setError('Error al consultar el backend: ' + err.message);
      setLoading(false);
    }
  };

  return (
    <div className="main-flex-layout">
      <div className="container">
        <h1>Multi-Agent-RAG-LangGraph</h1>
        <QuestionForm onSubmit={handleAsk} loading={loading} />
        {error && <div style={{ color: 'red', margin: 10 }}>{error}</div>}
        <AnswerBox answer={answer} />
        <LogViewer log={log} />
      </div>
      <div className="side-image-box">
        <img src="/static/advanced_build_graph.png" alt="Multi-Agent Graph" className="side-image" />
      </div>
    </div>
  );
}

export default App;
