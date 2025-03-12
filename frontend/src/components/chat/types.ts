export interface Message {
  id?: string;
  role: "user" | "assistant";
  content: string;
  feedback?: {
    rating?: boolean;
    feedbackText?: string;
    category?: string;
  };
  timestamp?: string;
  metadata?: {
    [key: string]: any;
    sources?: string[];
    used_chunks?: Array<{
      content: string;
      section: string;
      source: string;
    }>;
    generation_metrics?: {
      total_time_seconds?: number;
      llm_time_seconds?: number;
      context_retrieval_time_seconds?: number;
      response_length?: number;
      context_length?: number;
    };
    retrieval_analytics?: {
      num_context_chunks?: number;
      num_series_chunks?: number;
      num_non_series_chunks?: number;
      series_titles?: string[];
    };
    query_analysis?: {
      main_topic?: string;
      is_process_query?: boolean;
    };
    rag_provider?: string;
    response_to?: string;
    client_id?: string;
    temperature?: number;
    request_time?: string;
    is_first_message?: boolean;
  };
}

export interface ChatHistoryItem {
  id: string;
  title: string;
  timestamp: string;
}

export interface Theme {
  name: string;
  label: string;
  icon: string;
}

export interface FeedbackCategory {
  value: string;
  label: string;
  description: string;
}
