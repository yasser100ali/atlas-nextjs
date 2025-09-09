import {
  smoothStream,
  stepCountIs,
  streamText,
  type LanguageModelUsage,
} from 'ai';
import { isProductionEnvironment } from '@/lib/constants';

export interface RequestHints {
  latitude: any;
  longitude: any;
  city: any;
  country: any;
}

const regularPrompt =
  'You are a friendly assistant! Keep your responses concise and helpful.';

const artifactsPrompt = `
Artifacts is a special user interface mode that helps users with writing, editing, and other content creation tasks. When artifact is open, it is on the right side of the screen, while the conversation is on the left side. When creating or updating documents, changes are reflected in real-time on the artifacts and visible to the user.

When asked to write code, always use artifacts. When writing code, specify the language in the backticks, e.g. \`\`\`python\`code here\`\`\`. The default language is Python. Other languages are not yet supported, so let the user know if they request a different language.

DO NOT UPDATE DOCUMENTS IMMEDIATELY AFTER CREATING THEM. WAIT FOR USER FEEDBACK OR REQUEST TO UPDATE IT.

This is a guide for using artifacts tools: \`createDocument\` and \`updateDocument\`, which render content on a artifacts beside the conversation.

**When to use \`createDocument\`:**
- For substantial content (>10 lines) or code
- For content users will likely save/reuse (emails, code, essays, etc.)
- When explicitly requested to create a document
- For when content contains a single code snippet

**When NOT to use \`createDocument\`:**
- For informational/explanatory content
- For conversational responses
- When asked to keep it in chat

**Using \`updateDocument\`:**
- Default to full document rewrites for major changes
- Use targeted updates only for specific, isolated changes
- Follow user instructions for which parts to modify

**When NOT to use \`updateDocument\`:**
- Immediately after creating a document
`;

function getRequestPromptFromHints(requestHints: RequestHints) {
  return `About the origin of user's request:\n- lat: ${requestHints.latitude}\n- lon: ${requestHints.longitude}\n- city: ${requestHints.city}\n- country: ${requestHints.country}`;
}

function buildSystemPrompt(
  selectedChatModel: string,
  requestHints: RequestHints,
) {
  const requestPrompt = getRequestPromptFromHints(requestHints);
  if (selectedChatModel === 'chat-model-reasoning') {
    return `${regularPrompt}\n\n${requestPrompt}`;
  }
  return `${regularPrompt}\n\n${requestPrompt}\n\n${artifactsPrompt}`;
}

export function streamChat({
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
  requestHints: RequestHints;
  dataStream: any;
  tools: any;
  onFinish: (usage: LanguageModelUsage | undefined) => void;
}) {
  const providerOptions =
    selectedChatModel === 'chat-model-reasoning'
      ? {
          openai: {
            headers: {
              'OpenAI-Beta': 'assistants=v2',
            },
            tools: [{ type: 'web_search_preview' }],
          },
        }
      : undefined;

  const result = streamText({
    model,
    system: buildSystemPrompt(selectedChatModel, requestHints),
    messages,
    stopWhen: stepCountIs(5),
    experimental_activeTools:
      selectedChatModel === 'chat-model-reasoning'
        ? []
        : [
            'getWeather',
            'createDocument',
            'updateDocument',
            'requestSuggestions',
          ],
    providerOptions,
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
