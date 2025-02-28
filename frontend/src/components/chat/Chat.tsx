import { useState, useRef, useEffect, useCallback, memo } from "react";
import axios from "axios";
import { Toaster, toast } from "react-hot-toast";
import { FiCpu } from "react-icons/fi";
import { AnimatePresence } from "framer-motion";
import { PulseLoader } from "react-spinners";
import { v4 as uuidv4 } from "uuid";

// Import components
import { ChatMessage } from "./index";
import { ChatInput } from "./index";
import { ChatSidebar } from "./index";
import { FeedbackModal } from "./index";

// Import types
import { Message, ChatHistoryItem, FeedbackCategory } from "./types";
import { config as defaultConfig } from "../../config";

const FEEDBACK_CATEGORIES: FeedbackCategory[] = [
  {
    value: "incorrect",
    label: "Incorrect Information",
    description: "The response contains factually incorrect information",
  },
  {
    value: "incomplete",
    label: "Incomplete Answer",
    description: "The response is missing important information",
  },
  {
    value: "unclear",
    label: "Unclear Explanation",
    description: "The response is confusing or poorly explained",
  },
  {
    value: "not_helpful",
    label: "Not Helpful",
    description: "The response does not address my question",
  },
  {
    value: "outdated",
    label: "Outdated Information",
    description: "The information appears to be outdated",
  },
  {
    value: "other",
    label: "Other",
    description: "Other issues not listed above",
  },
];

interface ChatProps {
  config?: typeof defaultConfig;
}

const Chat: React.FC<ChatProps> = ({ config = defaultConfig }) => {
  const lastResponseRef = useRef<HTMLDivElement>(null);
  const loadingRef = useRef<HTMLDivElement>(null);

  // State for chat
  const [chatId, setChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "👋 Hi there! I'm here to assist you with your queries about the Aptos blockchain technology. Feel free to ask me anything about:\n\n" +
        "- Move programming language\n" +
        "- Smart contracts development\n" +
        "- Account management\n" +
        "- Transactions and gas fees\n" +
        "- Network architecture\n" +
        "- Token standards\n" +
        "- And much more!\n\n" +
        "What would you like to learn about? 🚀",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const copiedMessageIdRef = useRef<string | null>(null);
  const clientIdRef = useRef<string>("");
  const [chatHistories, setChatHistories] = useState<ChatHistoryItem[]>([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [feedbackModalOpen, setFeedbackModalOpen] = useState(false);
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(
    null,
  );
  const [selectedCategory, setSelectedCategory] = useState<string>("");
  const [feedbackText, setFeedbackText] = useState("");

  // Initialize client ID
  useEffect(() => {
    const storedClientId = localStorage.getItem("clientId");
    if (storedClientId) {
      clientIdRef.current = storedClientId;
    } else {
      const newClientId = uuidv4();
      localStorage.setItem("clientId", newClientId);
      clientIdRef.current = newClientId;
    }
  }, []);

  // Load chat histories when client ID is available
  useEffect(() => {
    const loadChatHistories = async () => {
      if (!clientIdRef.current) return;

      try {
        const response = await axios.get(
          `${config.apiBaseUrl}/chat/histories?client_id=${clientIdRef.current}`,
        );
        setChatHistories(response.data.histories);
      } catch (err) {
        console.error("Error loading chat histories:", err);
        toast.error("Failed to load chat histories");
      }
    };

    loadChatHistories();
  }, []);

  // Load chat history when chatId is available
  useEffect(() => {
    const loadChatHistory = async () => {
      if (!chatId) return;

      try {
        const response = await axios.get(
          `${config.apiBaseUrl}/chat/${chatId}/messages`,
        );
        setMessages(response.data.messages);

        // Add a small delay to ensure messages are rendered before scrolling
        setTimeout(() => {
          if (lastResponseRef.current) {
            lastResponseRef.current.scrollIntoView({
              behavior: "smooth",
              block: "start",
            });
          }
        }, 100);
      } catch (err) {
        console.error("Error loading chat history:", err);
        toast.error("Failed to load chat history");
        // If we can't load the history, reset to new chat
        setChatId(null);
        setMessages([
          {
            role: "assistant",
            content:
              "👋 Hi there! I'm here to assist you with your queries about the Aptos blockchain technology...",
          },
        ]);
      }
    };

    loadChatHistory();
  }, [chatId]);

  const scrollToLastResponse = () => {
    if (lastResponseRef.current) {
      lastResponseRef.current.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  };

  // Scroll to loading indicator when it appears
  useEffect(() => {
    if (isLoading && loadingRef.current) {
      loadingRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [isLoading]);

  const handleFeedback = async (
    messageId: string,
    rating: boolean,
    category?: string,
    feedbackText?: string,
  ) => {
    try {
      const message = messages.find((m) => m.id === messageId);
      if (!message || message.role !== "assistant") return;

      const userMessage =
        messages[messages.findIndex((m) => m.id === messageId) - 1];
      if (!userMessage || userMessage.role !== "user") return;

      // Update UI immediately
      setMessages((prev) =>
        prev.map((m) =>
          m.id === messageId
            ? { ...m, feedback: { rating, feedbackText, category } }
            : m,
        ),
      );

      // Send feedback to backend
      await axios.post(`${config.apiBaseUrl}/feedback`, {
        message_id: messageId,
        query: userMessage.content,
        response: message.content,
        rating,
        category,
        feedback_text: feedbackText,
        used_chunks: message.usedChunks,
        timestamp: new Date().toISOString(),
      });

      toast.success("Thank you for your feedback!", {
        icon: rating ? "👍" : "👎",
        duration: 2000,
      });
    } catch (err) {
      console.error("Error submitting feedback:", err);
      toast.error("Failed to submit feedback");
    }
  };

  const handleCopy = async (messageId: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      copiedMessageIdRef.current = messageId;
      toast.success("Copied to clipboard!", { duration: 2000 });
      // Reset the copied state after 2 seconds
      setTimeout(() => {
        copiedMessageIdRef.current = null;
      }, 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
      toast.error("Failed to copy text");
    }
  };

  const handleStreamResponse = async (response: Response) => {
    const reader = response.body?.getReader();
    if (!reader) return;

    setIsStreaming(true);
    let streamedContent = "";
    const streamingMessageId = uuidv4();

    try {
      // Add a temporary streaming message immediately
      setMessages((prev) => [
        ...prev,
        {
          id: streamingMessageId,
          role: "assistant",
          content: "",
        },
      ]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Convert the chunk to text and append to the streaming message
        const chunk = new TextDecoder().decode(value);
        streamedContent += chunk;

        // Update the last message with new content
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === streamingMessageId
              ? { ...msg, content: streamedContent }
              : msg,
          ),
        );
      }

      // For new chats, we need to get the chat ID from the backend
      if (!chatId) {
        try {
          // Get the latest chat history from the server
          const historyResponse = await axios.get(
            `${config.apiBaseUrl}/chat/latest?client_id=${clientIdRef.current}`,
          );

          const history = historyResponse.data;

          // Update the local chatId
          setChatId(history.id);

          // Update chat histories list
          setChatHistories((prev) => [
            {
              id: history.id,
              title: history.title,
              timestamp: history.timestamp,
            },
            ...prev,
          ]);

          console.log(`Set chat ID to ${history.id} for new chat`);
        } catch (error) {
          console.error("Error getting latest chat history:", error);
        }
      }
    } finally {
      reader.releaseLock();
      setIsStreaming(false);
    }
  };

  const handleSubmitMessage = useCallback(
    async (message: string) => {
      if (!message.trim() || !clientIdRef.current) return;

      const userMessage: Message = {
        role: "user",
        content: message,
        id: uuidv4(),
        timestamp: new Date().toISOString(),
      };

      // Add the user message to the UI immediately
      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);

      try {
        let response;
        if (!chatId) {
          // For new chats, we send the user message to create a new chat
          response = await fetch(`${config.apiBaseUrl}/chat/new/stream`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              messages: [userMessage],
              client_id: clientIdRef.current,
            }),
          });
        } else {
          // For existing chats, we send the user message to add to the existing chat
          // The backend will handle adding the message to the chat history
          response = await fetch(
            `${config.apiBaseUrl}/chat/${chatId}/message/stream`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                ...userMessage,
                client_id: clientIdRef.current,
                chat_id: chatId, // Explicitly include the chat_id to ensure it's used
              }),
            },
          );
        }

        if (!response.ok) {
          throw new Error("Failed to get response from the assistant");
        }

        await handleStreamResponse(response);

        toast.success("Response received!", {
          icon: "🤖",
          duration: 2000,
        });
        setTimeout(scrollToLastResponse, 100);
      } catch (err) {
        toast.error("Failed to get response from the assistant");
        console.error("Chat error:", err);
        // Remove the user message if the request failed
        setMessages((prev) => prev.slice(0, -1));
      } finally {
        setIsLoading(false);
      }
    },
    [chatId, config.apiBaseUrl],
  );

  const startNewChat = () => {
    setChatId(null);
    setMessages([
      {
        role: "assistant",
        content:
          "👋 Hi there! I'm here to assist you with your queries about the Aptos blockchain technology. Feel free to ask me anything about:\n\n" +
          "- Move programming language\n" +
          "- Smart contracts development\n" +
          "- Account management\n" +
          "- Transactions and gas fees\n" +
          "- Network architecture\n" +
          "- Token standards\n" +
          "- And much more!\n\n" +
          "What would you like to learn about? 🚀",
      },
    ]);
  };

  const handleDeleteChat = async (
    chatToDelete: ChatHistoryItem,
    e: React.MouseEvent,
  ) => {
    e.stopPropagation(); // Prevent chat selection when clicking delete

    // Show confirmation dialog
    if (
      !window.confirm(
        "Are you sure you want to delete this chat? This action cannot be undone.",
      )
    ) {
      return;
    }

    try {
      await axios.delete(
        `${config.apiBaseUrl}/chat/history/${chatToDelete.id}`,
      );

      // Remove from chat histories list
      setChatHistories((prev) =>
        prev.filter((chat) => chat.id !== chatToDelete.id),
      );

      // If the deleted chat was selected, reset to new chat
      if (chatId === chatToDelete.id) {
        startNewChat();
      }

      toast.success("Chat deleted successfully");
    } catch (err) {
      console.error("Error deleting chat:", err);
      toast.error("Failed to delete chat");
    }
  };

  const openFeedbackModal = (messageId: string) => {
    setSelectedMessageId(messageId);
    setSelectedCategory("");
    setFeedbackText("");
    setFeedbackModalOpen(true);
  };

  const submitNegativeFeedback = () => {
    if (!selectedMessageId) return;
    handleFeedback(selectedMessageId, false, selectedCategory, feedbackText);
    setFeedbackModalOpen(false);
    setSelectedMessageId(null);
    setSelectedCategory("");
    setFeedbackText("");
  };

  return (
    <div className={"h-full flex flex-col relative"} data-theme={"lofi"}>
      {/* Chat History Sidebar */}
      <ChatSidebar
        isSidebarOpen={isSidebarOpen}
        chatHistories={chatHistories}
        currentChatId={chatId}
        onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        onSelectChat={(id) => setChatId(id)}
        onDeleteChat={handleDeleteChat}
        onStartNewChat={startNewChat}
      />
      {/* Main Chat Area */}
      <div
        className={`p-2flex flex-col h-full ${isSidebarOpen ? "ml-64" : ""}`}
      >
        {/* Messages Area - Scrollable with fixed height */}
        <div
          className="flex-grow overflow-y-auto p-4 bg-base-100 rounded-lg"
          style={{
            minHeight: "100px",
            height: "calc(100% - 80px)",
            marginBottom: "60px",
          }}
        >
          <AnimatePresence>
            {messages.map((message, index) => (
              <ChatMessage
                key={index}
                message={message}
                isLastMessage={
                  index === messages.length - 1 && message.role === "assistant"
                }
                isStreaming={isStreaming}
                copiedMessageId={copiedMessageIdRef.current}
                onFeedback={handleFeedback}
                onOpenFeedbackModal={openFeedbackModal}
                onCopy={handleCopy}
                forwardedRef={
                  index === messages.length - 1 && message.role === "assistant"
                    ? lastResponseRef
                    : undefined
                }
              />
            ))}
          </AnimatePresence>

          {isLoading && (
            <div ref={loadingRef} className="flex justify-center">
              <div className="alert alert-info w-fit">
                <FiCpu className="animate-spin" />
                <span>Assistant is thinking...</span>
                <PulseLoader size={4} />
              </div>
            </div>
          )}
        </div>

        {/* Input Area - Positioned at bottom but within the container */}
        <div
          className={"absolute bottom-0 left-0 right-0 bg-base-100 p-2"}
          style={{
            zIndex: 10,
            marginLeft: isSidebarOpen ? "16rem" : "0",
            width: isSidebarOpen ? "calc(100% - 16rem)" : "100%",
          }}
        >
          <ChatInput isLoading={isLoading} onSubmit={handleSubmitMessage} />
        </div>
      </div>

      {/* Feedback Modal */}
      <FeedbackModal
        isOpen={feedbackModalOpen}
        selectedCategory={selectedCategory}
        feedbackText={feedbackText}
        categories={FEEDBACK_CATEGORIES}
        onCategoryChange={(e) => setSelectedCategory(e.target.value)}
        onFeedbackTextChange={(e) => setFeedbackText(e.target.value)}
        onSubmit={submitNegativeFeedback}
        onCancel={() => setFeedbackModalOpen(false)}
      />

      <Toaster position="bottom-right" />
    </div>
  );
};

export default memo(Chat);
