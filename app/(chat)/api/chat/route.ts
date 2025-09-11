import { auth } from '@/app/(auth)/auth';
import type { UserType } from '@/app/(auth)/auth';
import type { RequestHints } from '@/lib/ai/prompts';
import {
  createStreamId,
  deleteChatById,
  getChatById,
  getMessageCountByUserId,
  getMessagesByChatId,
  saveChat,
  saveMessages,
} from '@/lib/db/queries';
import {
  convertToUIMessages,
  convertToModelMessages,
  generateUUID,
} from '@/lib/utils';
import { generateTitleFromUserMessage } from '../../actions';
import { postRequestBodySchema } from './schema';
import type { PostRequestBody } from './schema';
import { geolocation } from '@vercel/functions';
import { createUIMessageStream, JsonToSseTransformStream } from 'ai';
import { getStreamContext } from '@/lib/stream-context';
import { ChatSDKError } from '@/lib/errors';
import type { ChatMessage } from '@/lib/types';
import type { ChatModel } from '@/lib/ai/models';
import type { VisibilityType } from '@/components/visibility-selector';

export const maxDuration = 60;

export async function POST(request: Request) {
  let requestBody: PostRequestBody;

  try {
    const json = await request.json();
    requestBody = postRequestBodySchema.parse(json);
  } catch (_) {
    return new ChatSDKError('bad_request:api').toResponse();
  }

  try {
    const {
      id,
      message,
      selectedChatModel,
      selectedVisibilityType,
    }: {
      id: string;
      message: ChatMessage;
      selectedChatModel: ChatModel['id'];
      selectedVisibilityType: VisibilityType;
    } = requestBody;

    const session = await auth();

    if (!session?.user) {
      return new ChatSDKError('unauthorized:chat').toResponse();
    }

    const userType: UserType = session.user.type;

    // Disable app-level per-day message limits
    // (previously compared against entitlementsByUserType[userType].maxMessagesPerDay)

    const chat = await getChatById({ id });

    if (!chat) {
      const title = await generateTitleFromUserMessage({
        message,
      });

      await saveChat({
        id,
        userId: session.user.id,
        title,
        visibility: selectedVisibilityType,
      });
    } else {
      if (chat.userId !== session.user.id) {
        return new ChatSDKError('forbidden:chat').toResponse();
      }
    }

    const messagesFromDb = await getMessagesByChatId({ id });
    const uiMessages = [...convertToUIMessages(messagesFromDb), message];

    const { longitude, latitude, city, country } = geolocation(request);

    const requestHints: RequestHints = {
      longitude,
      latitude,
      city,
      country,
    };

    await saveMessages({
      messages: [
        {
          chatId: id,
          id: message.id,
          role: 'user',
          parts: message.parts,
          attachments: [],
          createdAt: new Date(),
        },
      ],
    });

    const streamId = generateUUID();
    await createStreamId({ streamId, chatId: id });

    let finalUsage: any;

    const stream = createUIMessageStream({
      execute: async ({ writer: dataStream }) => {
        try {
          console.log('[chat:route] proxying to Python backend', {
            chatId: id,
            selectedChatModel,
          });

          // Proxy to Python backend
          const pythonResponse = await fetch('http://localhost:8000/api/chat', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              messages: convertToModelMessages(uiMessages),
              selectedChatModel,
              requestHints,
            }),
          });

          if (!pythonResponse.ok) {
            throw new Error(
              `Python backend error: ${pythonResponse.status} ${pythonResponse.statusText}`,
            );
          }

          // Stream the response from Python backend
          const reader = pythonResponse.body?.getReader();
          if (!reader) {
            throw new Error('No response body from Python backend');
          }

          const decoder = new TextDecoder();
          let buffer = '';
          let assistantId: string | null = null;

          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split('\n');

              // Keep the last incomplete line in buffer
              buffer = lines.pop() || '';

              for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;

                // Log every raw line for debugging purposes
                console.log('[chat:route] <- python:', trimmed.slice(0, 120));

                let jsonPayload: any = null;

                try {
                  const withoutPrefix = trimmed.startsWith('data: ')
                    ? trimmed.replace('data: ', '')
                    : trimmed;
                  jsonPayload = JSON.parse(withoutPrefix);
                } catch (err) {
                  console.warn(
                    '[chat:route] Could not parse line as JSON:',
                    trimmed,
                  );
                  continue;
                }

                // Forward to UI stream
                const { type } = jsonPayload;

                if (type === 'text-start') {
                  assistantId = generateUUID();
                  dataStream.write({
                    type: 'text-start',
                    id: assistantId,
                  } as any);
                } else if (type === 'text-delta') {
                  if (!assistantId) {
                    assistantId = generateUUID();
                    dataStream.write({
                      type: 'text-start',
                      id: assistantId,
                    } as any);
                  }
                  dataStream.write({
                    type: 'text-delta',
                    id: assistantId,
                    delta: jsonPayload.delta,
                  } as any);
                } else if (type === 'text-end') {
                  if (assistantId) {
                    dataStream.write({
                      type: 'text-end',
                      id: assistantId,
                    } as any);
                  }
                } else if (type === 'start-step') {
                  // Map python control event to UI control event
                  dataStream.write({ type: 'start' } as any);
                } else if (type === 'finish-step' || type === 'final') {
                  dataStream.write({ type: 'finish' } as any);
                  assistantId = null;
                } else {
                  // unknown data; forward raw
                  dataStream.write(jsonPayload as any);
                }
              }
            }
          } finally {
            reader.releaseLock();
          }
        } catch (error) {
          console.error(
            '[chat:route] Error proxying to Python backend:',
            error,
          );
          dataStream.write({
            type: 'text-delta',
            id: generateUUID(),
            delta: `\n[Error] Failed to connect to Python backend: ${
              (error as Error).message
            }\n`,
          });
          dataStream.write({ type: 'text-end' } as any);
          dataStream.write({ type: 'finish-step' } as any);
        }
      },
      generateId: generateUUID,
      onFinish: async ({ messages }) => {
        console.log('[chat:route] UI stream finished, saving messages', {
          count: messages.length,
        });
        await saveMessages({
          messages: messages.map((message) => ({
            id: message.id,
            role: message.role,
            parts: message.parts,
            createdAt: new Date(),
            attachments: [],
            chatId: id,
          })),
        });
      },
      onError: () => {
        console.error('[chat:route] UI stream error');
        return 'Oops, an error occurred!';
      },
    });

    const streamContext = getStreamContext();

    if (streamContext) {
      return new Response(
        await streamContext.resumableStream(streamId, () =>
          stream.pipeThrough(new JsonToSseTransformStream()),
        ),
      );
    } else {
      return new Response(stream.pipeThrough(new JsonToSseTransformStream()));
    }
  } catch (error) {
    if (error instanceof ChatSDKError) {
      return error.toResponse();
    }

    console.error('Unhandled error in chat API:', error);
    return new ChatSDKError('offline:chat').toResponse();
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');

  if (!id) {
    return new ChatSDKError('bad_request:api').toResponse();
  }

  const session = await auth();

  if (!session?.user) {
    return new ChatSDKError('unauthorized:chat').toResponse();
  }

  const chat = await getChatById({ id });

  if (chat.userId !== session.user.id) {
    return new ChatSDKError('forbidden:chat').toResponse();
  }

  const deletedChat = await deleteChatById({ id });

  return Response.json(deletedChat, { status: 200 });
}
