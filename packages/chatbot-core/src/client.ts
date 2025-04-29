import { ChatbotConfig, SendMessageOptions, Message, Chat } from './types';

export class ChatbotClient {
  private config: ChatbotConfig;

  constructor(config: ChatbotConfig) {
    this.config = config;
  }

  private async fetchWithAuth(url: string, options: RequestInit = {}): Promise<Response> {
    const headers = {
      'Content-Type': 'application/json',
      ...(this.config.apiKey && { Authorization: `Bearer ${this.config.apiKey}` }),
      ...options.headers,
    };

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    return response;
  }

  // Chat Management
  async createChat(): Promise<Chat> {
    const clientId = this.config.clientId;
    if (!clientId) {
      throw new Error('Client ID not found. Please ensure it is set in config.');
    }

    return {
      id: null,
      title: 'New Chat',
      timestamp: new Date().toISOString(),
      messages: [],
      metadata: {
        client_id: clientId,
        rag_provider: this.config.ragProvider,
      },
    };
  }

  async getChat(chatId: string): Promise<Chat> {
    const response = await this.fetchWithAuth(`${this.config.apiUrl}/api/chat/history/${chatId}`);
    return response.json();
  }

  async listChats(): Promise<Chat[]> {
    const clientId = this.config.clientId;
    if (!clientId) {
      throw new Error('Client ID not found. Please ensure it is set in config.');
    }

    const url = `${this.config.apiUrl}/api/chat/histories?client_id=${encodeURIComponent(clientId)}`;
    const response = await this.fetchWithAuth(url);
    const data = await response.json();

    if (!data || !data.histories) {
      throw new Error('Invalid response format from server');
    }

    return data.histories;
  }

  async deleteChat(chatId: string): Promise<void> {
    await this.fetchWithAuth(`${this.config.apiUrl}/api/chat/history/${chatId}`, {
      method: 'DELETE',
    });
  }

  async updateChatTitle(chatId: string, title: string): Promise<void> {
    await this.fetchWithAuth(`${this.config.apiUrl}/api/chat/history/${chatId}`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    });
  }

  // Message Operations
  async sendMessage(
    chatId: string | null,
    content: string,
    options?: SendMessageOptions,
  ): Promise<Response> {
    const clientId = this.config.clientId;
    if (!clientId) {
      throw new Error('Client ID not found. Please ensure it is set in config.');
    }

    const request = {
      content,
      client_id: clientId,
      role: 'user',
      temperature: 0.7,
      rag_provider: this.config.ragProvider || 'developer-docs',
      ...(options?.messageId && { message_id: options.messageId }),
      ...(chatId && { chat_id: chatId }),
    };

    console.log('Sending request:', request);

    const response = await this.fetchWithAuth(`${this.config.apiUrl}/api/message/stream`, {
      method: 'POST',
      body: JSON.stringify(request),
      headers: {
        Accept: 'text/plain', // Accept plain text for streaming
        'Content-Type': 'application/json',
      },
      signal: options?.signal,
    });

    // Check if the response is ok
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Error response:', errorText);
      throw new Error(
        `API request failed: ${response.status} ${response.statusText}\n${errorText}`,
      );
    }

    return response;
  }

  async getMessages(
    chatId: string,
    before?: string,
  ): Promise<{ messages: Message[]; hasMore: boolean }> {
    const url = new URL(`${this.config.apiUrl}/api/chat/history/${chatId}`);
    if (before) {
      url.searchParams.set('before', before);
    }

    const response = await this.fetchWithAuth(url.toString());
    return response.json();
  }

  async provideFeedback(messageId: string, feedback: 'positive' | 'negative'): Promise<void> {
    await this.fetchWithAuth(`${this.config.apiUrl}/api/feedback`, {
      method: 'POST',
      body: JSON.stringify({
        message_id: messageId,
        feedback,
        client_id: this.config.clientId,
      }),
    });
  }

  updateConfig(newConfig: Partial<ChatbotConfig>): void {
    this.config = {
      ...this.config,
      ...newConfig,
    };
  }

  getConfig(): ChatbotConfig {
    return { ...this.config };
  }
}
