import { Message, Chat } from '@aptos/chatbot-core';

export interface ChatWidgetProps {
  messages: Message[];
  isLoading?: boolean;
  isGenerating?: boolean;
  isTyping?: boolean;
  hasMoreMessages?: boolean;
  onSendMessage: (message: string) => void;
  onStopGenerating?: () => void;
  onLoadMore?: () => void;
  onCopyMessage?: (messageId: string) => void;
  onMessageFeedback?: (messageId: string, feedback: 'positive' | 'negative') => void;
  onNewChat?: () => void;
  className?: string;
  inputClassName?: string;
  messageClassName?: string;
  fastMode?: boolean;
  showSidebar?: boolean;
  // Chat history props
  chats?: Chat[];
  currentChatId?: string | null;
  onSelectChat?: (chatId: string) => void;
  onDeleteChat?: (chatId: string) => void;
  onUpdateChatTitle?: (chatId: string, title: string) => void;
  onToggleFastMode?: (enabled: boolean) => void;
}

export interface ChatContextState {
  fastMode?: boolean;
}
