import { type ReactElement } from 'react';
import { type ChatWidgetProps } from '@aptos/chatbot-react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { ChatSidebar } from './ChatSidebar';
import './styles/chatbot.css';

export function ChatWidget({
  messages,
  isLoading = false,
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
  detailedMode = false,
  showSidebar = false,
  // Chat history props
  chats = [],
  currentChatId,
  onSelectChat,
  onDeleteChat,
  onUpdateChatTitle,
  onToggleDetailedMode,
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
          detailedMode={detailedMode}
          onToggleDetailedMode={onToggleDetailedMode}
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
              detailed={detailedMode}
              onCopy={onCopyMessage}
              onFeedback={onMessageFeedback}
              className={messageClassName}
            />
          ))}
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
