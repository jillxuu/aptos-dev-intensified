import { useContext } from 'react';
import { ChatbotContext, ChatContextState } from '../context/ChatbotContext';

export function useChatbot(): ChatContextState {
  const context = useContext(ChatbotContext);
  if (!context) {
    throw new Error('useChatbot must be used within a ChatbotProvider');
  }
  return context;
}

export function useChatHistory(): Pick<
  ChatContextState,
  'chats' | 'createNewChat' | 'selectChat' | 'deleteChat' | 'updateChatTitle'
> {
  const context = useContext(ChatbotContext);
  if (!context) {
    throw new Error('useChatHistory must be used within a ChatbotProvider');
  }
  return {
    chats: context.chats,
    createNewChat: context.createNewChat,
    selectChat: context.selectChat,
    deleteChat: context.deleteChat,
    updateChatTitle: context.updateChatTitle,
  };
}
