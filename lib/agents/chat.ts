import {
  smoothStream,
  stepCountIs,
  streamText,
  type LanguageModelUsage,
} from 'ai';
import { isProductionEnvironment } from '../constants';

export async function streamChat({
  model,
  messages,
  selectedChatModel,
  requestHints,
  dataStream,
  tools,
  onFinish,
  chatId,
}: {
  model: any;
  messages: any;
  selectedChatModel: string;
  requestHints: { latitude: any; longitude: any; city: any; country: any };
  dataStream: any;
  tools: any;
  onFinish: (usage: LanguageModelUsage | undefined) => void;
  chatId: string;
}) {
  const buildSystemPrompt = () => {
    const req = `About the origin of user's request:
- lat: ${requestHints.latitude}
- lon: ${requestHints.longitude}
- city: ${requestHints.city}
- country: ${requestHints.country}`;
    const regular =
      'You are a friendly assistant! Keep your responses concise and helpful.';
    return `${regular}\n\n${req}`;
  };

  // ===== Route A: Python FastAPI backend for “reasoning” =====
  if (selectedChatModel === 'chat-model-reasoning') {
    const backend =
      process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';

    const toText = (content: any) =>
      typeof content === 'string'
        ? content
        : (content || [])
            .filter((p: any) => p.type === 'text')
            .map((p: any) => p.text)
            .join('');

    const pyMessages = messages.map((m: any) => ({
      role: m.role,
      content: toText(m.content),
    }));

    try {
      const response = await fetch(`${backend}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: pyMessages, chatId }),
      });

      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let previous = '';

      dataStream.write({ type: 'text-start' });

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const evt = JSON.parse(line);
            if (evt.event === 'final') {
              const text = evt.response || '';
              const delta = text.slice(previous.length);
              if (delta) {
                dataStream.write({ type: 'text-delta', delta });
              }
              previous = text;
            }
          } catch (err) {
            // Ignore malformed lines
          }
        }
      }

      dataStream.write({ type: 'text-end' });
      onFinish(undefined);
    } catch (error) {
      dataStream.write({
        type: 'text-delta',
        delta: `\n[agent error] ${error}`,
      });
      dataStream.write({ type: 'text-end' });
      onFinish(undefined);
    }
    return;
  }

  // ===== Route B: your existing Vercel AI SDK path =====
  const result = streamText({
    model,
    system: buildSystemPrompt(),
    messages,
    stopWhen: stepCountIs(5),
    experimental_activeTools:
      selectedChatModel === 'chat-model'
        ? [
            'getWeather',
            'createDocument',
            'updateDocument',
            'requestSuggestions',
          ]
        : [],
    providerOptions: undefined,
    experimental_transform: smoothStream({ chunking: 'word' }),
    tools,
    experimental_telemetry: {
      isEnabled: isProductionEnvironment,
      functionId: 'stream-text',
    },
    onFinish: ({ usage }) => {
      onFinish(usage);
      dataStream.write({ type: 'data-usage', data: usage });
    },
  });

  return result;
}
