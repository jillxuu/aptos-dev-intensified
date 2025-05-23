import { type ReactElement } from 'react';
import { type ChatWidgetProps } from '@aptos/chatbot-react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { ChatSidebar } from './ChatSidebar';
import './styles/chatbot.css';

export function ChatWidget({
  messages,
  isLoading = false,
  isGenerating = false,
  isTyping = false,
  hasMoreMessages = false,
  onSendMessage,
  onStopGenerating,
  onLoadMore,
  onCopyMessage,
  onMessageFeedback,
  onNewChat,
  className = '',
  inputClassName = '',
  messageClassName = '',
  fastMode = false,
  showSidebar = false,
  // Chat history props
  chats = [],
  currentChatId,
  onSelectChat,
  onDeleteChat,
  onUpdateChatTitle,
  onToggleFastMode,
}: ChatWidgetProps): ReactElement {
  return (
    <div className={`chat-widget ${className}`}>
      {showSidebar && (
        <ChatSidebar
          chats={chats}
          currentChatId={currentChatId}
          onNewChat={onNewChat}
          onSelectChat={onSelectChat}
          onDeleteChat={onDeleteChat}
          onUpdateChatTitle={onUpdateChatTitle}
          fastMode={fastMode}
          onToggleFastMode={onToggleFastMode}
          className="chat-sidebar"
        />
      )}
      <div className="chat-main">
        <div className="chat-messages">
          {hasMoreMessages && (
            <button onClick={onLoadMore} className="load-more-button" disabled={isLoading}>
              Load More
            </button>
          )}
          {messages.map(message => (
            <ChatMessage
              key={message.id}
              message={message}
              detailed={fastMode}
              onCopy={onCopyMessage}
              onFeedback={onMessageFeedback}
              className={messageClassName}
            />
          ))}
          {isGenerating && !isTyping && (
            <div className="chat-message assistant thinking">
              <div className="message-content">
                <div className="thinking-indicator">AI is thinking...</div>
              </div>
            </div>
          )}
        </div>
        <ChatInput
          onSend={onSendMessage}
          onStop={onStopGenerating}
          isLoading={isLoading}
          className={inputClassName}
        />
      </div>
    </div>
  );
}
