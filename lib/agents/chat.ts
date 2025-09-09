import {
  smoothStream,
  stepCountIs,
  streamText,
  type LanguageModelUsage,
} from 'ai';
import { isProductionEnvironment } from '../constants';

// ⬇️ Agents SDK
import { run, system, user } from '@openai/agents';
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

  // ===== Route A: Agents SDK for “reasoning” =====
  if (selectedChatModel === 'chat-model-reasoning') {
    const sys = buildSystemPrompt();
    const lastUser =
      messages
        ?.slice()
        .reverse()
        .find((m: any) => m.role === 'user')?.content ?? '';

    // Streaming tokens from the Agent to your existing dataStream
    const stream = await run(researchAgent, [system(sys), user(lastUser)], {
      stream: true,
    });

    // Use the text stream for simpler handling
    const textStream = stream.toTextStream();

    try {
      // Signal UI that assistant text is starting
      dataStream.write({ type: 'text-start' });
      for await (const chunk of textStream) {
        // UI expects { type: 'text-delta', delta: string }
        dataStream.write({ type: 'text-delta', delta: chunk });
      }
      // Close the assistant text block
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
