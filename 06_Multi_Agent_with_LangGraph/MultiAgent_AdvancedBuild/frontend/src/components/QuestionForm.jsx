import React, { useState } from 'react';

export default function QuestionForm({ onSubmit, loading }) {
  const [question, setQuestion] = useState('');
  const [apiKey, setApiKey] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!question || !apiKey) return;
    onSubmit({ question, apiKey });
  };

  return (
    <form onSubmit={handleSubmit} className="question-form">
      <div>
        <label>Question:</label>
        <textarea
          value={question}
          onChange={e => setQuestion(e.target.value)}
          rows={3}
          required
          placeholder="Type your question for the multi-agent system..."
        />
      </div>
      <div>
        <label>OpenAI API Key:</label>
        <input
          type="password"
          value={apiKey}
          onChange={e => setApiKey(e.target.value)}
          required
          placeholder="sk-..."
        />
      </div>
      <button type="submit" disabled={loading}>
        {loading ? 'Asking...' : 'Ask'}
      </button>
    </form>
  );
} 