import { type Message, type Chat } from '@aptos/chatbot-core';

export interface ChatMessageProps {
  message: Message;
  detailed?: boolean;
  fastMode?: boolean;
  onCopy?: (messageId: string) => void;
  onFeedback?: (messageId: string, feedback: 'positive' | 'negative') => void;
  className?: string;
}

export interface ChatInputProps {
  onSend: (message: string) => void;
  onStop?: () => void;
  isLoading?: boolean;
  placeholder?: string;
  className?: string;
}

export interface ChatSidebarProps {
  chats?: Chat[];
  currentChatId?: string;
  onNewChat?: () => void;
  onSelectChat?: (chatId: string) => void;
  onDeleteChat?: (chatId: string) => void;
  onUpdateChatTitle?: (chatId: string, title: string) => void;
  fastMode?: boolean;
  onToggleFastMode?: (enabled: boolean) => void;
  className?: string;
}

export interface ChatWidgetProps {
  messages: Message[];
  isLoading?: boolean;
  isGenerating?: boolean;
  isTyping?: boolean;
  hasMoreMessages?: boolean;
  fastMode?: boolean;
  showSidebar?: boolean;
  onSendMessage: (message: string) => void;
  onStopGenerating?: () => void;
  onLoadMore?: () => void;
  onCopyMessage?: (messageId: string) => void;
  onMessageFeedback?: (messageId: string, feedback: 'positive' | 'negative') => void;
  onNewChat?: () => void;
  className?: string;
  inputClassName?: string;
  messageClassName?: string;
  chats?: Chat[];
  currentChatId?: string;
  onSelectChat?: (chatId: string) => void;
  onDeleteChat?: (chatId: string) => void;
  onUpdateChatTitle?: (chatId: string, title: string) => void;
  onToggleFastMode?: (enabled: boolean) => void;
}
