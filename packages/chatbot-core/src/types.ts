export enum RagProvider {
  DEVELOPER_DOCS = 'developer-docs',
  APTOS_LEARN = 'aptos-learn',
}

export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: string;
  feedback?: 'positive' | 'negative';
}

export interface Chat {
  id: string | null;
  title: string;
  timestamp: string;
  lastMessage?: string;
  messages: Message[];
  metadata?: {
    client_id: string;
    rag_provider?: RagProvider;
    [key: string]: any;
  };
}

export interface ChatbotConfig {
  apiKey: string;
  apiUrl: string;
  clientId?: string;
  ragProvider?: RagProvider;
  githubRepo?: string;
  fastMode?: boolean;
}

// API request types
export interface ChatMessageRequest {
  content: string;
  client_id: string;
  message_id?: string;
  chat_id?: string;
  role?: 'user' | 'assistant';
  rag_provider?: RagProvider;
  temperature?: number;
}

// API response types
export interface ChatHistoriesResponse {
  histories: Chat[];
  totalCount: number;
}

export interface StatusResponse {
  success: boolean;
  message: string;
}

export type SendMessageOptions = {
  signal?: AbortSignal;
  onProgress?: (content: string) => void;
  messageId?: string;
};

export interface ChatResponse {
  message: Message;
  isComplete: boolean;
}

export interface ChatHistory {
  messages: Message[];
  hasMore: boolean;
}
