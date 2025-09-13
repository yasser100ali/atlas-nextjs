import { PreviewMessage, ThinkingMessage } from './message';
import { Greeting } from './greeting';
import { memo, useEffect, useRef } from 'react';
import type { Vote } from '@/lib/db/schema';
import equal from 'fast-deep-equal';
import type { UseChatHelpers } from '@ai-sdk/react';
import { useMessages } from '@/hooks/use-messages';
import type { ChatMessage } from '@/lib/types';
import { useDataStream } from './data-stream-provider';
import { Conversation, ConversationContent } from './elements/conversation';
import { ArrowDownIcon } from 'lucide-react';

interface MessagesProps {
  chatId: string;
  status: UseChatHelpers<ChatMessage>['status'];
  votes: Array<Vote> | undefined;
  messages: ChatMessage[];
  setMessages: UseChatHelpers<ChatMessage>['setMessages'];
  regenerate: UseChatHelpers<ChatMessage>['regenerate'];
  isReadonly: boolean;
  isArtifactVisible: boolean;
  selectedModelId: string;
}

function PureMessages({
  chatId,
  status,
  votes,
  messages,
  setMessages,
  regenerate,
  isReadonly,
  isArtifactVisible,
  selectedModelId,
}: MessagesProps) {
  const {
    containerRef: messagesContainerRef,
    endRef: messagesEndRef,
    isAtBottom,
    scrollToBottom,
    hasSentMessage,
  } = useMessages({
    chatId,
    status,
  });

  useDataStream();

  // Track previous message content for continuous scrolling during streaming
  const previousContentRef = useRef<string>('');

  // Continuous auto-scroll as AI types
  useEffect(() => {
    if (status === 'streaming' && messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      const currentContent = JSON.stringify(lastMessage.parts);

      // Check if content has actually changed
      if (currentContent !== previousContentRef.current) {
        const isFirstContent = previousContentRef.current === '';
        previousContentRef.current = currentContent;

        // Auto-scroll when content changes during streaming
        requestAnimationFrame(() => {
          const container = messagesContainerRef.current;
          if (container) {
            container.scrollTo({
              top: container.scrollHeight,
              // Use smooth for the first bit of content, then auto for continuous typing
              behavior: isFirstContent ? 'smooth' : 'auto',
            });
          }
        });
      }
    } else if (status !== 'streaming') {
      // Reset when not streaming
      previousContentRef.current = '';
    }
  }, [messages, status, messagesContainerRef]);

  // Initial smooth scroll when new prompt is submitted
  useEffect(() => {
    if (status === 'submitted') {
      // Use a small delay to ensure the new message is rendered
      setTimeout(() => {
        const container = messagesContainerRef.current;
        if (container) {
          container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth',
          });
        }
      }, 50); // Small delay for smooth transition
    }
  }, [status, messagesContainerRef]);

  return (
    <div
      ref={messagesContainerRef}
      className="overflow-y-scroll flex-1 touch-pan-y overscroll-behavior-contain -webkit-overflow-scrolling-touch"
      style={{ overflowAnchor: 'none' }}
    >
      <Conversation className="flex flex-col gap-4 px-2 py-4 mx-auto min-w-0 max-w-4xl md:gap-6 md:px-4">
        <ConversationContent className="flex flex-col gap-4 md:gap-6">
          {messages.length === 0 && <Greeting />}

          {messages.map((message, index) => (
            <PreviewMessage
              key={message.id}
              chatId={chatId}
              message={message}
              isLoading={
                status === 'streaming' && messages.length - 1 === index
              }
              vote={
                votes
                  ? votes.find((vote) => vote.messageId === message.id)
                  : undefined
              }
              setMessages={setMessages}
              regenerate={regenerate}
              isReadonly={isReadonly}
              requiresScrollPadding={
                hasSentMessage && index === messages.length - 1
              }
              isArtifactVisible={isArtifactVisible}
            />
          ))}

          {status === 'submitted' &&
            messages.length > 0 &&
            messages[messages.length - 1].role === 'user' &&
            selectedModelId !== 'chat-model-reasoning' && <ThinkingMessage />}

          <div
            ref={messagesEndRef}
            className="shrink-0 min-w-[24px] min-h-[24px]"
          />
        </ConversationContent>
      </Conversation>

      {!isAtBottom && (
        <button
          className="absolute bottom-40 left-1/2 z-10 p-2 rounded-full border shadow-lg transition-colors -translate-x-1/2 bg-background hover:bg-muted"
          onClick={() => scrollToBottom('smooth')}
          type="button"
          aria-label="Scroll to bottom"
        >
          <ArrowDownIcon className="size-4" />
        </button>
      )}
    </div>
  );
}

export const Messages = memo(PureMessages, (prevProps, nextProps) => {
  if (prevProps.isArtifactVisible && nextProps.isArtifactVisible) return true;

  if (prevProps.status !== nextProps.status) return false;
  if (prevProps.selectedModelId !== nextProps.selectedModelId) return false;
  if (prevProps.messages.length !== nextProps.messages.length) return false;
  if (!equal(prevProps.messages, nextProps.messages)) return false;
  if (!equal(prevProps.votes, nextProps.votes)) return false;

  return false;
});
