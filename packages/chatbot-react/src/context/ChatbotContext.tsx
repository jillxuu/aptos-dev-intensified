import React, { createContext, useState, useCallback, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { ChatbotClient, Message, ChatbotConfig, Chat } from '@aptos/chatbot-core';

export interface ChatContextState {
  // Connection State
  isLoading: boolean;
  error: Error | null;

  // Chat State
  currentChatId: string | null;
  chats: Chat[];
  messages: Message[];
  isTyping: boolean;
  hasMoreMessages: boolean;
  detailedMode: boolean;

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
  setDetailedMode: (enabled: boolean) => void;
}

const DEFAULT_CONTEXT: ChatContextState = {
  isLoading: false,
  error: null,
  currentChatId: null,
  chats: [],
  messages: [],
  isTyping: false,
  hasMoreMessages: false,
  detailedMode: false,
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
  setDetailedMode: () => {},
};

export const ChatbotContext = createContext<ChatContextState>(DEFAULT_CONTEXT);

interface ChatbotProviderProps {
  config: ChatbotConfig;
  children: React.ReactNode;
  onError?: (error: Error) => void;
}

export const ChatbotProvider: React.FC<ChatbotProviderProps> = ({
  config: initialConfig,
  children,
  onError,
}) => {
  // Client instance
  const clientRef = useRef(new ChatbotClient(initialConfig));

  // State
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [chats, setChats] = useState<Chat[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [hasMoreMessages, setHasMoreMessages] = useState(false);
  const [detailedMode, setDetailedMode] = useState(initialConfig.detailedMode ?? false);
  const [isOpen, setIsOpen] = useState(false);
  const [config, setConfig] = useState(initialConfig);

  // Abort controller for cancelling requests
  const abortControllerRef = useRef<AbortController | null>(null);

  // Configuration
  const updateConfig = useCallback((newConfig: Partial<ChatbotConfig>) => {
    console.log('Updating config:', newConfig);
    setConfig(prev => {
      const updated = { ...prev, ...newConfig };
      clientRef.current.updateConfig(updated);
      return updated;
    });
  }, []);

  // Chat operations
  const loadChats = useCallback(async () => {
    console.log('loadChats called, attempting to fetch chats');
    try {
      console.log('Current client config:', clientRef.current.getConfig());
      const response = await clientRef.current.listChats();
      console.log('Received chats from backend:', response);
      setChats(response);
    } catch (err) {
      console.error('Error loading chats:', err);
      const error = err as Error;
      setError(error);
      onError?.(error);
      // Reset chats on error
      setChats([]);
    }
  }, [onError]);

  // Initialize clientId
  useEffect(() => {
    console.log('Initializing clientId');
    const storedClientId = localStorage.getItem('clientId');
    if (storedClientId) {
      console.log('Found stored clientId:', storedClientId);
      updateConfig({ clientId: storedClientId });
    } else {
      const newClientId = uuidv4();
      console.log('Generated new clientId:', newClientId);
      localStorage.setItem('clientId', newClientId);
      updateConfig({ clientId: newClientId });
    }
  }, [updateConfig]);

  // Load chats when clientId is available
  useEffect(() => {
    console.log('Chat loading effect triggered. Current config:', config);
    if (config.clientId) {
      console.log('ClientId available, loading chats for:', config.clientId);
      loadChats();
    } else {
      console.log('No clientId available yet, skipping chat load');
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
        const messageId = `msg-${uuidv4()}`;
        const message: Message = {
          id: messageId,
          content,
          role: 'user',
          timestamp: new Date().toISOString(),
        };

        // If no current chat, create one first
        let chatId = currentChatId;
        if (!chatId) {
          const newChat = await clientRef.current.createChat();
          chatId = newChat.id;
          setCurrentChatId(chatId);
          await loadChats();
        }

        // Add user message to the current chat
        setMessages(prev => [...prev, message]);

        // Create a new abort controller for this request
        const abortController = new AbortController();
        abortControllerRef.current = abortController;

        // Add assistant's response placeholder
        const assistantMessage: Message = {
          id: `msg-${uuidv4()}`,
          content: '',
          role: 'assistant',
          timestamp: new Date().toISOString(),
        };
        setMessages(prev => [...prev, assistantMessage]);

        // Send message and get streaming response
        const response = await clientRef.current.sendMessage(chatId, content, {
          messageId,
          signal: abortController.signal,
        });

        // Process streaming response
        if (response.body) {
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let responseText = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            // Decode chunk and update assistant message
            const chunk = decoder.decode(value, { stream: true });
            responseText += chunk;

            setMessages(prev => {
              const lastMessage = prev[prev.length - 1];
              if (lastMessage.role === 'assistant') {
                return [...prev.slice(0, -1), { ...lastMessage, content: responseText }];
              }
              return prev;
            });
          }

          // After message is fully processed, refresh the chat list
          await loadChats();
        }
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') {
          // Ignore abort errors
          return;
        }
        const error = err as Error;
        setError(error);
        onError?.(error);
        // Remove the last two messages on error
        setMessages(prev => prev.slice(0, -2));
      } finally {
        abortControllerRef.current = null;
        setIsLoading(false);
        setIsTyping(false);
      }
    },
    [currentChatId, onError, config.clientId, loadChats],
  );

  // Chat management
  const createNewChat = useCallback(async () => {
    setIsLoading(true);
    try {
      const chat = await clientRef.current.createChat();
      setCurrentChatId(chat.id);
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
        setMessages(chat.messages);
      }
    },
    [chats],
  );

  // History management
  const loadPreviousMessages = useCallback(async () => {
    if (!currentChatId || isLoading || !hasMoreMessages) return;

    setIsLoading(true);
    try {
      const firstMessage = messages[0];
      const response = await clientRef.current.getMessages(currentChatId, firstMessage?.id);
      setMessages(prev => [...response.messages, ...prev]);
      setHasMoreMessages(response.hasMore);
    } catch (err) {
      const error = err as Error;
      setError(error);
      onError?.(error);
    } finally {
      setIsLoading(false);
    }
  }, [currentChatId, isLoading, hasMoreMessages, messages, onError]);

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
  const setDetailedModeCallback = useCallback(
    (enabled: boolean) => {
      setDetailedMode(enabled);
      updateConfig({ detailedMode: enabled });
    },
    [updateConfig],
  );

  return (
    <ChatbotContext.Provider
      value={{
        isLoading,
        error,
        currentChatId,
        chats,
        messages,
        isTyping,
        hasMoreMessages,
        detailedMode,
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
        setDetailedMode: setDetailedModeCallback,
      }}
    >
      {children}
    </ChatbotContext.Provider>
  );
};
