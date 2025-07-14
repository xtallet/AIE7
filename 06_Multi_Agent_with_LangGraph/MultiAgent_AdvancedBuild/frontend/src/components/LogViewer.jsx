import React from 'react';

export default function LogViewer({ log }) {
  if (!log || log.length === 0) return null;
  return (
    <div className="log-viewer">
      <h3>Execution Log</h3>
      <pre>
        {log.join('\n')}
      </pre>
    </div>
  );
} 