import React from 'react';

export default function AnswerBox({ answer }) {
  if (!answer) return null;
  return (
    <div className="answer-box">
      <h3>Answer</h3>
      <div>{answer}</div>
    </div>
  );
} 