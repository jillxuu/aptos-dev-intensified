import { useState, useRef, useEffect, useCallback, memo } from "react";
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
// Import API service
import { apiService } from "../../services";

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
  // Track the source of chatId changes to determine when to load history
  const [chatIdSource, setChatIdSource] = useState<
    "initial" | "selection" | "stream"
  >("initial");

  // Initialize client ID and check URL for chat ID
  useEffect(() => {
    const storedClientId = localStorage.getItem("clientId");
    if (storedClientId) {
      clientIdRef.current = storedClientId;
    } else {
      const newClientId = uuidv4();
      localStorage.setItem("clientId", newClientId);
      clientIdRef.current = newClientId;
    }

    // Check URL for chat ID
    const urlParams = new URLSearchParams(window.location.search);
    const urlChatId = urlParams.get("chat");
    if (urlChatId) {
      console.log(`Found chat ID in URL: ${urlChatId}`);
      // Set source to 'selection' for URL-based chat loading
      setChatIdSource("selection");
      setChatId(urlChatId);
    }
  }, []);

  // Load chat histories when client ID is available
  useEffect(() => {
    if (clientIdRef.current) {
      loadChatHistories();
    }
  }, [clientIdRef.current]);

  const loadChatHistories = async () => {
    try {
      setIsLoading(true);
      console.log("Loading chat histories");

      const response = await apiService.getChatHistories(clientIdRef.current);

      if (response && response.data && response.data.histories) {
        console.log(
          `Received ${response.data.histories.length} chat histories`,
        );

        // Ensure no duplicates in the histories
        const uniqueHistories = [];
        const seenIds = new Set();

        for (const history of response.data.histories) {
          if (!seenIds.has(history.id)) {
            seenIds.add(history.id);
            uniqueHistories.push(history);
          }
        }

        if (uniqueHistories.length !== response.data.histories.length) {
          console.warn(
            `Removed ${response.data.histories.length - uniqueHistories.length} duplicate histories`,
          );
        }

        setChatHistories(uniqueHistories);
      } else {
        console.warn("No chat histories received or invalid response format");
      }
    } catch (err) {
      console.error("Error loading chat histories:", err);
      toast.error("Failed to load chat histories");
    } finally {
      setIsLoading(false);
    }
  };

  // Load chat history when chatId is available and it's not from streaming
  useEffect(() => {
    if (chatId && chatIdSource !== "stream") {
      console.log(
        `Loading chat history for chatId: ${chatId}, source: ${chatIdSource}`,
      );
      loadChatHistory();
    } else {
      console.log(
        `Skipping chat history load for chatId: ${chatId}, source: ${chatIdSource}`,
      );
    }
  }, [chatId, chatIdSource]);

  const loadChatHistory = async () => {
    try {
      setIsLoading(true);
      console.log(`Fetching messages for chat: ${chatId}`);

      // Ensure chatId is not null before making the API call
      if (!chatId) {
        console.warn("Cannot load chat history: chatId is null");
        setIsLoading(false);
        return;
      }

      // Log the API endpoint being called for debugging
      console.log(`Calling API endpoint: /chat/history/${chatId}`);

      const response = await apiService.getChatMessages(chatId);

      if (response && response.data) {
        // The response contains the entire chat history object, not just messages
        const chatHistory = response.data;

        if (chatHistory.messages && Array.isArray(chatHistory.messages)) {
          console.log(
            `Received ${chatHistory.messages.length} messages for chat: ${chatId}`,
          );
          setMessages(chatHistory.messages);
        } else {
          console.warn(`No messages found in chat history for chat: ${chatId}`);
        }
      } else {
        console.warn(`No chat history received for chat: ${chatId}`);
      }

      // Add a small delay to ensure messages are rendered before scrolling
      setTimeout(() => {
        scrollToLastResponse();
      }, 100);
    } catch (err) {
      console.error(`Error loading chat history for ${chatId}:`, err);
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
      await apiService.submitFeedback({
        message_id: messageId,
        chat_id: chatId,
        rating,
        category,
        feedback_text: feedbackText,
        usedChunks: responseMessage.metadata?.used_chunks,
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

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const handleStreamResponse = async (response: Response) => {
    if (!response.body) {
      toast.error("Failed to get response stream");
      return;
    }

    // Extract chat_id from response headers
    const headerChatId = response.headers.get("X-Chat-ID");
    let extractedChatId = chatId; // Initialize with current chatId

    if (headerChatId) {
      extractedChatId = headerChatId;
      console.log(`Extracted chat ID from header: ${extractedChatId}`);

      // If this is a new chat (chatId was null), update the chatId
      if (!chatId) {
        console.log(`Setting new chat ID: ${extractedChatId}`);
        // Set the source before updating chatId
        setChatIdSource("stream");

        // Update the chat ID
        setChatId(extractedChatId);

        // Update the URL with the new chat ID
        window.history.replaceState(null, "", `?chat=${extractedChatId}`);
      }
    } else {
      console.warn("No chat ID found in response headers");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let responseText = "";
    let responseId = uuidv4();

    console.log(`Starting stream with response ID: ${responseId}`);

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

        // Decode the chunk and add to response text
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

      console.log("Stream completed, updating chat history");

      // No need to get the chat history since we already have the chat ID
      if (extractedChatId) {
        // Add this chat to the histories list, but prevent duplicates
        loadChatHistories();
      }
    } catch (err) {
      console.error("Error reading stream:", err);
      toast.error("Error receiving response");
    }
  };

  const handleSendMessage = useCallback(
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
        // Use the unified endpoint for both new chats and adding messages to existing chats
        const requestBody: any = {
          content: message,
          client_id: clientIdRef.current,
          role: "user",
          message_id: userMessage.id,
          temperature: 0.7,
        };

        // If we have a chat ID, include it to add to an existing chat
        if (chatId) {
          requestBody.chat_id = chatId;
        }

        console.log(
          `Sending message to unified endpoint, chat_id: ${chatId || "new"}`,
        );

        // Use the API service for streaming
        const response = await apiService.sendMessageStream(requestBody);

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        await handleStreamResponse(response);
      } catch (err) {
        toast.error("Failed to get response from the assistant");
        console.error("Chat error:", err);
        // Remove the user message if the request failed
        setMessages((prev) => prev.slice(0, -1));
      } finally {
        setIsLoading(false);
      }
    },
    [chatId, handleStreamResponse],
  );

  const startNewChat = () => {
    setChatIdSource("selection");
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
      await apiService.deleteChatHistory(chatToDelete.id);

      // Remove from the list
      setChatHistories((prev) =>
        prev.filter((chat) => chat.id !== chatToDelete.id),
      );

      // If the deleted chat was the active one, reset to a new chat
      if (chatId === chatToDelete.id) {
        setChatIdSource("selection");
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

  // Update the ChatSidebar onSelectChat handler
  const handleSelectChat = (id: string) => {
    setChatIdSource("selection");
    setChatId(id);
  };

  return (
    <div className={"h-full flex flex-col relative"} data-theme={"lofi"}>
      {/* Chat History Sidebar */}
      <ChatSidebar
        isSidebarOpen={isSidebarOpen}
        chatHistories={chatHistories}
        currentChatId={chatId}
        onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        onSelectChat={handleSelectChat}
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
          <ChatInput isLoading={isLoading} onSubmit={handleSendMessage} />
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
