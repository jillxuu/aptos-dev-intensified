// Export the main plugin component
export { AptosChatbotPlugin } from "./AptosChatbotPlugin";

// Export sub-components in case users want to use them directly
export { Modal } from "./Modal";
export { TriggerButton } from "./TriggerButton";

// Re-export types from the chat components that might be needed
export type {
  Message,
  ChatHistoryItem,
  Theme,
  FeedbackCategory,
} from "../chat/types";
