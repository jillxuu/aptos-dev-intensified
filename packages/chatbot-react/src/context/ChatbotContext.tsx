import React, { createContext, useState, useCallback, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { ChatbotClient, Message, ChatbotConfig, Chat } from '@aptos/chatbot-core';

export interface ChatContextState {
  // Connection State
  isLoading: boolean;
  error: Error | null;
  isLoadingChats: boolean;
  isLoadingMoreMessages: boolean;

  // Chat State
  currentChatId: string | null;
  chats: Chat[];
  messages: Message[];
  isGenerating: boolean;
  isTyping: boolean;
  hasMoreMessages: boolean;
  fastMode: boolean;

  // Modal State
  isOpen: boolean;
  openChat: () => void;
  closeChat: () => void;

  // Chat Management
  createNewChat: () => Promise<void>;
  selectChat: (chatId: string) => void;
  deleteChat: (chatId: string) => Promise<void>;
  updateChatTitle: (chatId: string, title: string) => Promise<void>;

  // Message Operations
  sendMessage: (content: string) => Promise<void>;
  stopGenerating: () => void;
  retryLastMessage: () => Promise<void>;
  clearHistory: () => void;
  copyMessage: (messageId: string) => void;
  provideFeedback: (messageId: string, feedback: 'positive' | 'negative') => Promise<void>;

  // History Management
  loadPreviousMessages: () => Promise<void>;
  loadChats: () => Promise<void>;

  // Configuration
  config: ChatbotConfig;
  updateConfig: (newConfig: Partial<ChatbotConfig>) => void;
  setFastMode: (enabled: boolean) => void;
}

const DEFAULT_CONTEXT: ChatContextState = {
  isLoading: false,
  error: null,
  isLoadingChats: false,
  isLoadingMoreMessages: false,
  currentChatId: null,
  chats: [],
  messages: [],
  isGenerating: false,
  isTyping: false,
  hasMoreMessages: false,
  fastMode: false,
  isOpen: false,
  openChat: () => {},
  closeChat: () => {},
  createNewChat: async () => {},
  selectChat: () => {},
  deleteChat: async () => {},
  updateChatTitle: async () => {},
  sendMessage: async () => {},
  stopGenerating: () => {},
  retryLastMessage: async () => {},
  clearHistory: () => {},
  copyMessage: () => {},
  provideFeedback: async () => {},
  loadPreviousMessages: async () => {},
  loadChats: async () => {},
  config: { apiKey: '', apiUrl: '' },
  updateConfig: () => {},
  setFastMode: () => {},
};

export const ChatbotContext = createContext<ChatContextState>(DEFAULT_CONTEXT);

interface ChatbotProviderProps {
  config: ChatbotConfig;
  children: React.ReactNode;
  onError?: (error: Error) => void;
}

// Add this helper function before the ChatbotProvider component
const updateChatMessages = (chats: Chat[], chatId: string, messages: Message[]): Chat[] => {
  return chats.map(chat => {
    if (chat.id === chatId) {
      return { ...chat, messages, lastMessage: messages[messages.length - 1]?.content };
    }
    return chat;
  });
};

export const ChatbotProvider: React.FC<ChatbotProviderProps> = ({
  config: initialConfig,
  children,
  onError,
}) => {
  // Client instance
  const clientRef = useRef(new ChatbotClient(initialConfig));

  // State
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingChats, setIsLoadingChats] = useState(false);
  const [isLoadingMoreMessages, setIsLoadingMoreMessages] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [chats, setChats] = useState<Chat[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [hasMoreMessages, setHasMoreMessages] = useState(false);
  const [fastMode, setFastMode] = useState(initialConfig.fastMode ?? false);
  const [isOpen, setIsOpen] = useState(false);
  const [config, setConfig] = useState(initialConfig);

  // Abort controller for cancelling requests
  const abortControllerRef = useRef<AbortController | null>(null);

  // Configuration
  const updateConfig = useCallback((newConfig: Partial<ChatbotConfig>) => {
    setConfig(prev => {
      const updated = { ...prev, ...newConfig };
      clientRef.current.updateConfig(updated);
      return updated;
    });
  }, []);

  // Chat operations
  const loadChats = useCallback(async () => {
    try {
      setIsLoadingChats(true);
      const response = await clientRef.current.listChats();
      // Check for duplicate chat IDs
      setChats(response);
    } catch (err) {
      console.error('Error loading chats:', err);
      const error = err as Error;
      setError(error);
      onError?.(error);
      setChats([]);
    } finally {
      setIsLoadingChats(false);
    }
  }, [onError]);

  // Initialize clientId
  useEffect(() => {
    const storedClientId = localStorage.getItem('clientId');
    if (storedClientId) {
      updateConfig({ clientId: storedClientId });
    } else {
      const newClientId = uuidv4();
      localStorage.setItem('clientId', newClientId);
      updateConfig({ clientId: newClientId });
    }
  }, [updateConfig]);

  // Load chats when clientId is available
  useEffect(() => {
    if (config.clientId) {
      loadChats();
    }
  }, [config.clientId, loadChats]);

  // Modal controls
  const openChat = useCallback(() => setIsOpen(true), []);
  const closeChat = useCallback(() => setIsOpen(false), []);

  // Chat operations
  const sendMessage = useCallback(
    async (content: string) => {
      const clientId = config.clientId;
      if (!clientId) {
        console.error('No clientId available yet');
        setError(new Error('Client ID not initialized. Please try again.'));
        onError?.(new Error('Client ID not initialized. Please try again.'));
        return;
      }

      try {
        setIsGenerating(true);
        setIsLoading(true);

        const messageId = `msg-${uuidv4()}`;
        const message: Message = {
          id: messageId,
          content,
          role: 'user',
          timestamp: new Date().toISOString(),
        };

        // Update both messages and chats state
        setMessages(prev => [...prev, message]);
        if (currentChatId) {
          setChats(prev => updateChatMessages(prev, currentChatId, [...messages, message]));
        }

        const abortController = new AbortController();
        abortControllerRef.current = abortController;

        const response = await clientRef.current.sendMessage(currentChatId, content, {
          messageId,
          signal: abortController.signal,
        });

        const headerChatId = response.headers.get('X-Chat-ID');
        if (headerChatId && !currentChatId) {
          setCurrentChatId(headerChatId);
          await loadChats();
        }

        if (response.body) {
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let responseText = '';
          let assistantMessageId = `msg-${uuidv4()}`;
          let isFirstChunk = true;

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            responseText += chunk;

            if (isFirstChunk) {
              setIsGenerating(false);
              setIsTyping(true);
              const assistantMessage: Message = {
                id: assistantMessageId,
                content: responseText,
                role: 'assistant',
                timestamp: new Date().toISOString(),
              };
              setMessages(prev => [...prev, assistantMessage]);
              if (currentChatId) {
                setChats(prev =>
                  updateChatMessages(prev, currentChatId, [...messages, message, assistantMessage]),
                );
              }
              isFirstChunk = false;
            } else {
              setMessages(prev => {
                const lastMessage = prev[prev.length - 1];
                if (lastMessage.id === assistantMessageId) {
                  const updatedMessages = [
                    ...prev.slice(0, -1),
                    { ...lastMessage, content: responseText },
                  ];
                  if (currentChatId) {
                    setChats(prevChats =>
                      updateChatMessages(prevChats, currentChatId, updatedMessages),
                    );
                  }
                  return updatedMessages;
                }
                return prev;
              });
            }
          }

          await loadChats();
        }
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') {
          return;
        }
        const error = err as Error;
        setError(error);
        onError?.(error);
        setMessages(prev => prev.slice(0, -1));
        if (currentChatId) {
          setChats(prev => updateChatMessages(prev, currentChatId, messages.slice(0, -1)));
        }
      } finally {
        abortControllerRef.current = null;
        setIsLoading(false);
        setIsTyping(false);
        setIsGenerating(false);
      }
    },
    [currentChatId, messages, onError, config.clientId, loadChats],
  );

  // Chat management
  const createNewChat = useCallback(async () => {
    setIsLoading(true);
    try {
      // Let the backend assign the chat ID through the first message
      setCurrentChatId(null);
      setMessages([]);
      setHasMoreMessages(false);

      // Load updated chat list to refresh sidebar
      await loadChats();
    } catch (err) {
      const error = err as Error;
      setError(error);
      onError?.(error);
    } finally {
      setIsLoading(false);
    }
  }, [onError, loadChats]);

  const selectChat = useCallback(
    (chatId: string) => {
      setCurrentChatId(chatId);
      const chat = chats.find(c => c.id === chatId);
      if (chat) {
        setMessages(chat.messages || []);
      }
    },
    [chats],
  );

  // History management
  const loadPreviousMessages = useCallback(async () => {
    if (!currentChatId || isLoading || !hasMoreMessages || isLoadingMoreMessages) return;

    try {
      setIsLoadingMoreMessages(true);
      const firstMessage = messages[0];
      const response = await clientRef.current.getMessages(currentChatId, firstMessage?.id);
      setMessages(prev => [...response.messages, ...prev]);
      setHasMoreMessages(response.hasMore);
    } catch (err) {
      const error = err as Error;
      setError(error);
      onError?.(error);
    } finally {
      setIsLoadingMoreMessages(false);
    }
  }, [currentChatId, isLoading, hasMoreMessages, messages, onError, isLoadingMoreMessages]);

  // Chat operations
  const deleteChat = useCallback(
    async (chatId: string) => {
      try {
        await clientRef.current.deleteChat(chatId);
        if (currentChatId === chatId) {
          setCurrentChatId(null);
          setMessages([]);
        }
        await loadChats();
      } catch (err) {
        const error = err as Error;
        setError(error);
        onError?.(error);
      }
    },
    [currentChatId, onError, loadChats],
  );

  const updateChatTitle = useCallback(
    async (chatId: string, title: string) => {
      try {
        await clientRef.current.updateChatTitle(chatId, title);
        await loadChats();
      } catch (err) {
        const error = err as Error;
        setError(error);
        onError?.(error);
      }
    },
    [onError, loadChats],
  );

  // Message operations
  const copyMessage = useCallback(
    (messageId: string) => {
      const message = messages.find(m => m.id === messageId);
      if (message) {
        navigator.clipboard.writeText(message.content);
      }
    },
    [messages],
  );

  const provideFeedback = useCallback(
    async (messageId: string, feedback: 'positive' | 'negative') => {
      try {
        await clientRef.current.provideFeedback(messageId, feedback);
        setMessages(prev => prev.map(msg => (msg.id === messageId ? { ...msg, feedback } : msg)));
      } catch (err) {
        const error = err as Error;
        setError(error);
        onError?.(error);
      }
    },
    [onError],
  );

  // Chat operations
  const stopGenerating = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsTyping(false);
    setIsLoading(false);
  }, []);

  const retryLastMessage = useCallback(async () => {
    const lastUserMessage = [...messages].reverse().find(m => m.role === 'user');
    if (lastUserMessage) {
      // Remove the last assistant message if it exists
      setMessages(prev => {
        const lastMessage = prev[prev.length - 1];
        if (lastMessage.role === 'assistant') {
          return prev.slice(0, -1);
        }
        return prev;
      });
      await sendMessage(lastUserMessage.content);
    }
  }, [messages, sendMessage]);

  const clearHistory = useCallback(() => {
    setMessages([]);
    setHasMoreMessages(false);
  }, []);

  // Configuration
  const setFastModeCallback = useCallback(
    (enabled: boolean) => {
      setFastMode(enabled);
      updateConfig({ fastMode: enabled });
    },
    [updateConfig],
  );

  return (
    <ChatbotContext.Provider
      value={{
        isLoading,
        isLoadingChats,
        isLoadingMoreMessages,
        error,
        currentChatId,
        chats,
        messages,
        isGenerating,
        isTyping,
        hasMoreMessages,
        fastMode,
        isOpen,
        openChat,
        closeChat,
        createNewChat,
        selectChat,
        deleteChat,
        updateChatTitle,
        sendMessage,
        stopGenerating,
        retryLastMessage,
        clearHistory,
        copyMessage,
        provideFeedback,
        loadPreviousMessages,
        loadChats,
        config,
        updateConfig,
        setFastMode: setFastModeCallback,
      }}
    >
      {children}
    </ChatbotContext.Provider>
  );
};
