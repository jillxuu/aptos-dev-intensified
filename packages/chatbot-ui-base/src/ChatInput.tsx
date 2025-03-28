import { type ReactElement, useState, useRef, useEffect } from 'react';
import type { FormEvent, KeyboardEvent, ChangeEvent } from 'react';
import { type ChatInputProps } from '@aptos/chatbot-react';

export function ChatInput({
  onSend,
  onStop,
  isLoading = false,
  placeholder = 'Type a message...',
  className = '',
}: ChatInputProps): ReactElement {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [message]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!message.trim() || isLoading) return;

    onSend(message.trim());
    setMessage('');
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value);
  };

  return (
    <form onSubmit={handleSubmit} className={`chat-input ${className}`}>
      <textarea
        ref={textareaRef}
        value={message}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={isLoading}
        rows={1}
        className="input-textarea"
      />
      {isLoading ? (
        <button type="button" onClick={onStop} className="stop-button">
          Stop
        </button>
      ) : (
        <button type="submit" disabled={!message.trim()} className="send-button">
          Send
        </button>
      )}
    </form>
  );
}
