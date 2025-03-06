export interface RAGProvider {
  name: string;
  description: string;
}

export interface Config {
  apiBaseUrl: string;
  ragProvider?: string;
  ragConfig?: Record<string, any>;
}

export const config: Config = {
  apiBaseUrl: import.meta.env.VITE_API_URL || "http://localhost:8000/api",
  ragProvider: import.meta.env.VITE_RAG_PROVIDER || undefined,
};
