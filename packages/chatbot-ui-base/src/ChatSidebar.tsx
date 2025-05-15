import { type ReactElement } from 'react';
import { type ChatSidebarProps } from '@aptos/chatbot-react';

export function ChatSidebar({
  onNewChat,
  chats = [],
  currentChatId,
  onSelectChat,
  onDeleteChat,
  onUpdateChatTitle,
  fastMode,
  onToggleFastMode,
  className = '',
}: ChatSidebarProps): ReactElement {
  return (
    <div className={`chat-sidebar ${className}`}>
      <div className="chat-list">
        {chats.map(chat => (
          <div
            key={chat.id ?? 'new'}
            className={`chat-item ${chat.id === currentChatId ? 'active' : ''}`}
          >
            <button
              className="chat-button"
              onClick={() => chat.id && onSelectChat?.(chat.id)}
              disabled={!chat.id}
            >
              <span className="chat-title">{chat.title}</span>
              <span className="chat-time">{new Date(chat.timestamp).toLocaleTimeString()}</span>
            </button>
            <div className="chat-actions">
              <button
                className="edit-button"
                onClick={() => {
                  if (!chat.id) return;
                  const newTitle = prompt('Enter new title:', chat.title);
                  if (newTitle && onUpdateChatTitle) {
                    onUpdateChatTitle(chat.id, newTitle);
                  }
                }}
                disabled={!chat.id}
              >
                Edit
              </button>
              <button
                className="delete-button"
                onClick={() => {
                  if (!chat.id) return;
                  if (confirm('Are you sure you want to delete this chat?') && onDeleteChat) {
                    onDeleteChat(chat.id);
                  }
                }}
                disabled={!chat.id}
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        <div className="settings">
          <label className="detailed-mode-toggle">
            <input
              type="checkbox"
              checked={fastMode}
              onChange={e => onToggleFastMode?.(e.target.checked)}
            />
            Fast Mode
          </label>
        </div>
        <button onClick={onNewChat} className="new-chat-button">
          New Chat
        </button>
      </div>
    </div>
  );
}
