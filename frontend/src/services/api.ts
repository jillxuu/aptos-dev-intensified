import axios, { AxiosResponse } from "axios";
import { config } from "../config";

// Create an axios instance with default configuration
const apiClient = axios.create({
  baseURL: config.apiBaseUrl,
  headers: {
    "Content-Type": "application/json",
  },
});

// Add request interceptor to add RAG provider header if specified
apiClient.interceptors.request.use((requestConfig) => {
  if (config.ragProvider) {
    requestConfig.headers["X-RAG-Provider"] = config.ragProvider;
  }
  return requestConfig;
});

// API endpoints
export const API_ENDPOINTS = {
  // Chat endpoints
  CHAT_HISTORIES: (clientId: string) => `/chat/histories?client_id=${clientId}`,
  CHAT_MESSAGES: (chatId: string) => `/chat/history/${chatId}`,
  SEND_MESSAGE: "/message/stream",
  DELETE_CHAT: (chatId: string) => `/chat/history/${chatId}`,

  // Feedback endpoint
  FEEDBACK: "/feedback",

  // RAG provider endpoints
  GITHUB_INITIALIZE: "/rag/provider/github/initialize",
};

// API service methods
export const apiService = {
  // Chat methods
  getChatHistories: (clientId: string): Promise<AxiosResponse> => {
    console.log(`Fetching chat histories for client: ${clientId}`);
    return apiClient.get(API_ENDPOINTS.CHAT_HISTORIES(clientId));
  },

  getChatMessages: (chatId: string): Promise<AxiosResponse> => {
    const endpoint = API_ENDPOINTS.CHAT_MESSAGES(chatId);
    console.log(`Fetching chat messages from endpoint: ${endpoint}`);
    return apiClient.get(endpoint);
  },

  // Streaming message endpoint (returns fetch Response for streaming)
  sendMessageStream: async (data: any): Promise<Response> => {
    const url = `${config.apiBaseUrl}${API_ENDPOINTS.SEND_MESSAGE}`;

    // Add RAG provider if specified
    if (config.ragProvider) {
      data.rag_provider = config.ragProvider;
    }

    return fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/plain", // Accept plain text for streaming
        ...(config.ragProvider ? { "X-RAG-Provider": config.ragProvider } : {}),
      },
      body: JSON.stringify(data),
    });
  },

  // Feedback method
  submitFeedback: (feedbackData: any): Promise<AxiosResponse> => {
    return apiClient.post(API_ENDPOINTS.FEEDBACK, feedbackData);
  },

  // GitHub RAG provider methods
  initializeGitHubRepo: (repoData: {
    repo_url: string;
    branch: string;
  }): Promise<AxiosResponse> => {
    return apiClient.post(API_ENDPOINTS.GITHUB_INITIALIZE, repoData);
  },

  // Delete chat history
  deleteChatHistory: (chatId: string): Promise<AxiosResponse> => {
    return apiClient.delete(API_ENDPOINTS.DELETE_CHAT(chatId));
  },
};

export default apiService;
