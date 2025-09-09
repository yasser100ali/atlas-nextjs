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

  // ===== Route A: Agents SDK for “reasoning” =====
  if (selectedChatModel === 'chat-model-reasoning') {
    console.log('[streamChat] routing: reasoning (agents)');
    const sys = buildSystemPrompt();
    const lastUser =
      messages
        ?.slice()
        .reverse()
        .find((m: any) => m.role === 'user')?.content ?? '';

    // Streaming tokens from the Agent to your existing dataStream
    console.log('[streamChat] invoking researchAgent.run with string input');
    const combinedInput = `${sys}\n\n${lastUser}`;

    // Retry the streamed run to mitigate provider rate limits
    async function runWithRetry(attempt = 1): Promise<any> {
      try {
        return await run(researchAgent, combinedInput, { stream: true });
      } catch (error) {
        const maxAttempts = 3;
        const baseMs = 750;
        const shouldRetry =
          (error as any)?.code === 'rate_limit_exceeded' &&
          attempt < maxAttempts;
        if (!shouldRetry) throw error;
        const delay = baseMs * Math.pow(2, attempt - 1);
        console.warn('[streamChat] rate limited, retrying', { attempt, delay });
        await new Promise((r) => setTimeout(r, delay));
        return runWithRetry(attempt + 1);
      }
    }

    const stream = await runWithRetry();

    // Use the text stream for simpler handling
    console.log('[streamChat] toTextStream(nodeCompatible=true)');
    const textStream = stream.toTextStream({ compatibleWithNodeStreams: true });

    try {
      // Signal UI that a new step and assistant text are starting
      dataStream.write({ type: 'start-step' });
      dataStream.write({ type: 'text-start' });
      let chunkCount = 0;
      for await (const chunk of textStream) {
        chunkCount += 1;
        // UI expects { type: 'text-delta', delta: string }
        dataStream.write({ type: 'text-delta', delta: chunk });
      }
      // Close the assistant text block and step
      dataStream.write({ type: 'text-end' });
      dataStream.write({ type: 'finish-step' });
      // Signal the overall end so the UI assembles the assistant message
      dataStream.write({ type: 'final' });
      console.log('[streamChat] reasoning stream complete', { chunkCount });
      onFinish(undefined);
    } catch (error) {
      console.error('[streamChat] reasoning stream error', error);
      dataStream.write({
        type: 'text-delta',
        delta: `\n[agent error] ${error}`,
      });
      dataStream.write({ type: 'text-end' });
      dataStream.write({ type: 'finish-step' });
      dataStream.write({ type: 'final' });
      onFinish(undefined);
    }
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
