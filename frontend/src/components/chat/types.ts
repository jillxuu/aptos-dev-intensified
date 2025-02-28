export interface Message {
  id?: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  feedback?: {
    rating?: boolean;
    feedbackText?: string;
    category?: string;
  };
  usedChunks?: Array<{
    content: string;
    section: string;
    source: string;
  }>;
  timestamp?: string;
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
