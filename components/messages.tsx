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
  const isScrollingRef = useRef<boolean>(false);

  const NEAR_BOTTOM_PX = 120;     // how close counts as “near bottom”
  const REVEAL_GAP_PX = 24;       // how much hidden content below view triggers a reveal
  
  const isNearBottom = (el: HTMLElement) =>
    el.scrollHeight - el.scrollTop - el.clientHeight <= NEAR_BOTTOM_PX;
  
  const hasHiddenTail = (el: HTMLElement) =>
    el.scrollHeight - el.scrollTop - el.clientHeight > REVEAL_GAP_PX;
  

  const userNearBottomRef = useRef(true);

  useEffect(() => {
    const el = messagesContainerRef.current;
    if (!el) return;
  
    const onScroll = () => {
      userNearBottomRef.current = isNearBottom(el);
    };
  
    el.addEventListener('scroll', onScroll, { passive: true });
    // initialize
    onScroll();
    return () => el.removeEventListener('scroll', onScroll);
  }, [messagesContainerRef]);

  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;
  
    if (status === 'streaming' && messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      const currentContent = JSON.stringify(
        (lastMessage as any).parts ?? (lastMessage as any).content ?? lastMessage
      );
  
      if (currentContent !== previousContentRef.current) {
        previousContentRef.current = currentContent;
  
        // Only auto-reveal while streaming if the user is at/near the bottom.
        if (userNearBottomRef.current) {
          // If the tail is below the viewport, nudge down smoothly (never up)
          const target = container.scrollHeight - container.clientHeight;
          if (target > container.scrollTop && hasHiddenTail(container)) {
            // Large distances use instant to avoid lagging behind; small use smooth
            const distance = target - container.scrollTop;
            const behavior: ScrollBehavior = distance < 800 ? 'smooth' : 'auto';
            requestAnimationFrame(() => {
              container.scrollTo({ top: target, behavior });
            });
          }
        }
      }
    } else {
      previousContentRef.current = '';
      isScrollingRef.current = false;
    }
  }, [messages, status, messagesContainerRef]);
  

  // When user submits a prompt, scroll only if they are near the bottom (avoid jump-up-then-down)
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;
    if (status !== 'submitted') return;

    if (isNearBottom(container)) {
      const target = container.scrollHeight - container.clientHeight;
      requestAnimationFrame(() => {
        container.scrollTo({ top: target, behavior: 'smooth' });
      });
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
