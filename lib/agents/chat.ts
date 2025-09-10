import {
  smoothStream,
  stepCountIs,
  streamText,
  type LanguageModelUsage,
} from 'ai';
import { isProductionEnvironment } from '../constants';

// ⬇️ Direct OpenAI Responses API
import { researchAgent } from './researchAgent';

export async function streamChat({
  model,
  messages,
  selectedChatModel,
  requestHints,
  dataStream,
  tools,
  onFinish,
}: {
  model: any;
  messages: any;
  selectedChatModel: string;
  requestHints: { latitude: any; longitude: any; city: any; country: any };
  dataStream: any;
  tools: any;
  onFinish: (usage: LanguageModelUsage | undefined) => void;
}) {
  console.log('[streamChat] start', {
    selectedChatModel,
  });
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

  // ===== Route A: Direct OpenAI Responses API for "reasoning" =====
  if (selectedChatModel === 'chat-model-reasoning') {
    console.log('[streamChat] routing: reasoning (python backend)');
    const pythonBackendUrl = 'http://127.0.0.1:8000/api/chat';

    fetch(pythonBackendUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages, selectedChatModel, requestHints }),
    })
      .then(async (response) => {
        if (!response.body) {
          throw new Error('Python backend response has no body');
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        try {
          let buffer = '';
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split('\n\n');
            buffer = parts.pop() ?? ''; // The last part might be incomplete
            for (const part of parts) {
              if (part.startsWith('data: ')) {
                try {
                  const jsonData = part.substring(6);
                  if (jsonData.trim()) {
                    const data = JSON.parse(jsonData);
                    dataStream.write(data);
                  }
                } catch (e) {
                  console.error('Error parsing JSON from python stream:', e);
                }
              }
            }
          }
        } catch (error) {
          console.error('[streamChat] python stream error', error);
          dataStream.write({
            type: 'text-delta',
            delta: `\n[python backend error] ${error}`,
          });
          dataStream.write({ type: 'text-end' });
          dataStream.write({ type: 'finish-step' });
          dataStream.write({ type: 'final' });
        } finally {
          onFinish(undefined);
        }
      })
      .catch((error) => {
        console.error('[streamChat] python fetch error', error);
        dataStream.write({
          type: 'text-delta',
          delta: `\n[python backend error] ${error}`,
        });
        dataStream.write({ type: 'text-end' });
        dataStream.write({ type: 'finish-step' });
        dataStream.write({ type: 'final' });
        onFinish(undefined);
      });
    return;
  }

  // ===== Route B: your existing Vercel AI SDK path =====
  console.log('[streamChat] routing: regular (ai SDK)', {
    modelId: selectedChatModel,
  });
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
      console.log('[streamChat] regular onFinish', { usage });
      onFinish(usage);
      dataStream.write({ type: 'data-usage', data: usage });
    },
  });

  console.log('[streamChat] returning StreamTextResult for regular path');
  return result;
}
