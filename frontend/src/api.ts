import type { LogMessage } from './types';

export async function streamRequest<T>(
  url: string,
  body: any,
  onLog: (log: LogMessage) => void,
  onResult: (result: T) => void,
  onError: (error: string) => void
) {
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    if (!response.body) {
        throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const msg = JSON.parse(line);
          if (msg.type === 'result') {
            onResult(msg.content);
          } else if (msg.type === 'exception' || msg.type === 'error') {
            onError(msg.content);
          } else {
            // Assume it's a log message
            onLog(msg);
          }
        } catch (e) {
          console.error('Error parsing JSON line', line, e);
        }
      }
    }
  } catch (e) {
    onError(String(e));
  }
}

export async function getModels() {
    const response = await fetch('/api/models');
    if (!response.ok) {
        throw new Error('Failed to fetch models');
    }
    return response.json();
}
