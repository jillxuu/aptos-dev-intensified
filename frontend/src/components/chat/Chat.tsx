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

  // Get custom greeting based on RAG provider
  const getCustomGreeting = () => {
    const provider = config.ragProvider || "aptos";

    // Default Aptos greeting
    if (provider === "aptos") {
      return (
        "👋 Hi there! I'm your Aptos AI assistant. I can help you with:\n\n" +
        "- Move programming language\n" +
        "- Smart contracts development\n" +
        "- Account management\n" +
        "- Transactions and gas fees\n" +
        "- Network architecture\n" +
        "- Token standards\n" +
        "- And much more!\n\n" +
        "What would you like to learn about today? 🚀"
      );
    }

    // GitHub repository greeting
    if (provider === "github") {
      return (
        "👋 Hi there! I'm your AI assistant for this GitHub repository.\n\n" +
        "I can help you understand:\n" +
        "- Code structure and organization\n" +
        "- Implementation details\n" +
        "- Documentation and examples\n" +
        "- Project features and functionality\n" +
        "- And more!\n\n" +
        "What would you like to know about this repository? 🚀"
      );
    }

    // Generic greeting for other providers
    return (
      "👋 Hi there! I'm your AI assistant for this knowledge base.\n\n" +
      "I can help answer questions about the content and information in this knowledge source.\n\n" +
      "What would you like to know? 🚀"
    );
  };

  // State for chat
  const [chatId, setChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: getCustomGreeting(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const copiedMessageIdRef = useRef<string | null>(null);
  const clientIdRef = useRef<string>("");
  const [chatHistories, setChatHistories] = useState<ChatHistoryItem[]>([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [isStreaming, _setIsStreaming] = useState(false);
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

  // Function to make API requests with RAG provider header
  const makeApiRequest = async (url: string, method: string, data?: any) => {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    // Add RAG provider header if specified
    if (config.ragProvider) {
      headers["X-RAG-Provider"] = config.ragProvider;
    }

    try {
      if (method === "GET") {
        return await axios.get(url, { headers });
      } else if (method === "POST") {
        return await axios.post(url, data, { headers });
      } else if (method === "PUT") {
        return await axios.put(url, data, { headers });
      } else if (method === "DELETE") {
        return await axios.delete(url, { headers });
      }
    } catch (error) {
      console.error(`Error making ${method} request to ${url}:`, error);
      throw error;
    }
  };

  // Load chat histories when client ID is available
  useEffect(() => {
    if (clientIdRef.current) {
      loadChatHistories();
    }
  }, [clientIdRef.current]);

  const loadChatHistories = async () => {
    try {
      setIsLoading(true);
      const response = await makeApiRequest(
        `${config.apiBaseUrl}/chat/histories?client_id=${clientIdRef.current}`,
        "GET",
      );
      if (response) {
        setChatHistories(response.data.histories);
      }
    } catch (err) {
      console.error("Error loading chat histories:", err);
      toast.error("Failed to load chat histories");
    } finally {
      setIsLoading(false);
    }
  };

  // Load chat history when chatId is available
  useEffect(() => {
    if (chatId) {
      loadChatHistory();
    }
  }, [chatId]);

  const loadChatHistory = async () => {
    try {
      setIsLoading(true);
      const response = await makeApiRequest(
        `${config.apiBaseUrl}/chat/${chatId}/messages`,
        "GET",
      );
      if (response) {
        setMessages(response.data.messages);
      }

      // Add a small delay to ensure messages are rendered before scrolling
      setTimeout(() => {
        scrollToLastResponse();
      }, 100);
    } catch (err) {
      console.error("Error loading chat history:", err);
      toast.error("Failed to load chat history");
    } finally {
      setIsLoading(false);
    }
  };

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
      // Find the message that received feedback
      const responseMessage = messages.find(
        (msg) => msg.id === messageId && msg.role === "assistant",
      );

      if (!responseMessage) {
        console.error("Message not found for feedback");
        return;
      }

      // Submit feedback to the API
      await makeApiRequest(`${config.apiBaseUrl}/feedback`, "POST", {
        message_id: messageId,
        chat_id: chatId,
        rating,
        category,
        feedback_text: feedbackText,
        usedChunks: responseMessage.usedChunks,
        timestamp: new Date().toISOString(),
      });

      // Update the UI to show feedback was submitted
      const updatedMessages = [...messages];
      const msgToUpdate = updatedMessages.find((msg) => msg.id === messageId);
      if (msgToUpdate) {
        msgToUpdate.feedback = {
          rating,
          category,
          feedbackText,
        };
      }
      setMessages(updatedMessages);

      // Show success message
      toast.success(
        rating
          ? "Thanks for your positive feedback!"
          : "Thanks for your feedback. We'll use it to improve.",
      );

      // Close the feedback modal if it's open
      setFeedbackModalOpen(false);
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
    if (!response.body) {
      toast.error("Failed to get response stream");
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let responseText = "";
    let responseId = uuidv4();

    // Add the assistant message to the UI immediately
    setMessages((prevMessages) => [
      ...prevMessages,
      {
        role: "assistant",
        content: "",
        id: responseId,
        timestamp: new Date().toISOString(),
      },
    ]);

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Decode and append to the response text
        const chunk = decoder.decode(value, { stream: true });
        responseText += chunk;

        // Update the message in the UI
        setMessages((prevMessages) => {
          const updatedMessages = [...prevMessages];
          const lastIndex = updatedMessages.length - 1;
          if (lastIndex >= 0 && updatedMessages[lastIndex].id === responseId) {
            updatedMessages[lastIndex] = {
              ...updatedMessages[lastIndex],
              content: responseText,
            };
          }
          return updatedMessages;
        });

        // Scroll to the bottom as new content arrives
        if (lastResponseRef.current) {
          lastResponseRef.current.scrollIntoView({
            behavior: "smooth",
            block: "end",
          });
        }
      }

      // Get the chat history to update the URL and sidebar
      try {
        const historyResponse = await makeApiRequest(
          `${config.apiBaseUrl}/chat/latest?client_id=${clientIdRef.current}`,
          "GET",
        );

        if (historyResponse) {
          // Update the chat ID and title
          setChatId(historyResponse.data.id);

          // Update the URL with the new chat ID
          window.history.replaceState(
            null,
            "",
            `?chat=${historyResponse.data.id}`,
          );

          // Add this chat to the histories list
          setChatHistories((prev) => [historyResponse.data, ...prev]);
        }
      } catch (historyErr) {
        console.error("Error getting chat history:", historyErr);
      }
    } catch (err) {
      console.error("Error reading stream:", err);
      toast.error("Error receiving response");
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
        content: getCustomGreeting(),
      },
    ]);
  };

  const handleDeleteChat = async (
    chatToDelete: ChatHistoryItem,
    e: React.MouseEvent,
  ) => {
    e.stopPropagation();

    try {
      await makeApiRequest(
        `${config.apiBaseUrl}/chat/history/${chatToDelete.id}`,
        "DELETE",
      );

      // Remove from the list
      setChatHistories((prev) =>
        prev.filter((chat) => chat.id !== chatToDelete.id),
      );

      // If the deleted chat was the active one, reset to a new chat
      if (chatId === chatToDelete.id) {
        setChatId(null);
        setMessages([
          {
            role: "assistant",
            content: getCustomGreeting(),
            id: uuidv4(),
            timestamp: new Date().toISOString(),
          },
        ]);

        // Update the URL
        window.history.replaceState(null, "", window.location.pathname);
      }

      toast.success("Chat deleted");
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
