import { useState } from 'react';
import { ChatbotProvider, useChatbot } from '@aptos/chatbot-react';
import { ChatWidget } from '@aptos/chatbot-ui-base';
import { RagProvider } from '@aptos/chatbot-core';
import { Prism as SyntaxHighlighter, SyntaxHighlighterProps } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const PrismSyntaxHighlighter = SyntaxHighlighter as unknown as React.FC<SyntaxHighlighterProps>;

const basicExample = `// Basic Integration
import { ChatbotProvider } from '@aptos/chatbot-react';
import { ChatWidget } from '@aptos/chatbot-ui-tailwind';

function App() {
  return (
    <ChatbotProvider
      config={{
        apiKey: process.env.CHATBOT_API_KEY,
        apiUrl: process.env.CHATBOT_API_URL,
        detailedMode: false
      }}
    >
      <ChatWidget showSidebar />
    </ChatbotProvider>
  );
}`;

const customExample = `// Custom Implementation
import { useChatbot } from '@aptos/chatbot-react';

function CustomChatWidget() {
  const {
    messages,
    sendMessage,
    isLoading,
    detailedMode,
    chats,
    createNewChat,
    selectChat,
    deleteChat,
    updateChatTitle
  } = useChatbot();

  return (
    <div>
      {/* Your custom UI implementation */}
      <div className="chat-messages">
        {messages.map(message => (
          <div key={message.id}>
            {message.content}
          </div>
        ))}
      </div>
      
      <div className="chat-input">
        <input
          type="text"
          onKeyPress={e => {
            if (e.key === 'Enter') {
              sendMessage(e.target.value);
            }
          }}
        />
      </div>
    </div>
  );
}`;

const hookExample = `// Available Hooks and Types
import { useChatbot, useChatHistory } from '@aptos/chatbot-react';

// Core Types
interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: string;
  feedback?: 'positive' | 'negative';
}

interface Chat {
  id: string;
  title: string;
  timestamp: string;
  lastMessage?: string;
  messages: Message[];
}

// Hook Return Types
interface ChatContextState {
  // Chat State
  currentChatId: string | null;
  chats: Chat[];
  messages: Message[];
  isTyping: boolean;
  hasMoreMessages: boolean;
  detailedMode: boolean;

  // Chat Operations
  sendMessage: (content: string) => Promise<void>;
  createNewChat: () => Promise<void>;
  selectChat: (chatId: string) => void;
  deleteChat: (chatId: string) => Promise<void>;
  updateChatTitle: (chatId: string, title: string) => Promise<void>;
  
  // Message Operations
  stopGenerating: () => void;
  retryLastMessage: () => Promise<void>;
  copyMessage: (messageId: string) => void;
  provideFeedback: (messageId: string, feedback: 'positive' | 'negative') => Promise<void>;
  
  // History Management
  loadPreviousMessages: () => Promise<void>;
  loadChats: () => Promise<void>;
}`;

function ChatWidgetContainer() {
  const {
    messages = [],
    sendMessage,
    isLoading = false,
    hasMoreMessages = false,
    detailedMode = false,
    chats = [],
    currentChatId,
    stopGenerating,
    loadPreviousMessages,
    copyMessage,
    provideFeedback,
    createNewChat,
    selectChat,
    deleteChat,
    updateChatTitle,
    setDetailedMode,
  } = useChatbot();

  return (
    <ChatWidget
      messages={messages}
      onSendMessage={sendMessage}
      isLoading={isLoading}
      hasMoreMessages={hasMoreMessages}
      detailedMode={detailedMode}
      showSidebar={true}
      onStopGenerating={stopGenerating}
      onLoadMore={loadPreviousMessages}
      onCopyMessage={copyMessage}
      onMessageFeedback={provideFeedback}
      onNewChat={createNewChat}
      chats={chats}
      currentChatId={currentChatId || undefined}
      onSelectChat={selectChat}
      onDeleteChat={deleteChat}
      onUpdateChatTitle={updateChatTitle}
      onToggleDetailedMode={setDetailedMode}
      className="h-full"
    />
  );
}

interface CodeBlockProps {
  code: string;
  language?: string;
}

function CodeBlock({ code, language = 'typescript' }: CodeBlockProps) {
  return (
    <div className="rounded-lg overflow-hidden">
      <PrismSyntaxHighlighter
        style={vscDarkPlus}
        language={language}
        PreTag="div"
        className="rounded-lg overflow-x-auto"
        customStyle={{
          margin: 0,
          padding: '1rem',
          borderRadius: '0.5rem',
          fontSize: '0.9rem',
        }}
      >
        {code}
      </PrismSyntaxHighlighter>
    </div>
  );
}

const ragArchitectureExample = `// RAG Architecture Flow
┌──────────────┐     ┌──────────────┐     ┌───────────────────┐
│   User       │     │   Backend    │     │    Document       │
│   Query      │────>│   Server     │────>│    Processing     │
└──────────────┘     └──────────────┘     └───────────────────┘
                           │                        │
                           │                        ▼
                           │               ┌───────────────────┐
                           │               │ Text Chunking     │
                           │               │ ・Size: 512 tokens│
                           │               │ ・Overlap: 50     │
                           │               └───────────────────┘
                           │                        │
                           │                        ▼
                           │               ┌───────────────────┐
                           │               │ Topic Extraction  │
                           │               │ ・Keywords        │
                           │               │ ・Categories      │
                           │               └───────────────────┘
                           │                        │
                           │                        ▼
                           │               ┌───────────────────┐
                           │               │ Vector Store      │
                           │               │ ・Embeddings      │
                           │               │ ・Metadata        │
                           │               └───────────────────┘
                           │                        │
                           ▼                        │
                    ┌──────────────┐               │
                    │   Context    │<──────────────┘
                    │   Window     │
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐     ┌──────────────┐
                    │    LLM       │────>│  Generated   │
                    │   Model      │     │   Response   │
                    └──────────────┘     └──────────────┘`;

export default function App() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Aptos Chatbot Demo</h1>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="grid grid-cols-1 gap-8">
            {/* Introduction */}
            <section className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">Getting Started</h2>
              <p className="text-gray-600 mb-6">
                The Aptos Chatbot package provides a flexible and customizable chat interface with
                built-in support for chat history, message feedback, and detailed mode. Try out the
                demo by clicking the button below.
              </p>
              <button
                onClick={() => setIsOpen(true)}
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
              >
                Open Chat Demo
              </button>
            </section>

            {/* Basic Integration */}
            <section className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">Basic Integration</h2>
              <p className="text-gray-600 mb-4">
                Get started quickly with the pre-built UI components:
              </p>
              <CodeBlock code={basicExample} />
            </section>

            {/* Custom Implementation */}
            <section className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">Custom Implementation</h2>
              <p className="text-gray-600 mb-4">
                Build your own UI using the provided hooks and types:
              </p>
              <CodeBlock code={customExample} />
            </section>

            {/* RAG Architecture */}
            <section className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">RAG Architecture</h2>
              <p className="text-gray-600 mb-4">
                The Aptos Chatbot uses Retrieval Augmented Generation (RAG) to provide accurate,
                context-aware responses:
              </p>
              <div className="mb-6">
                <CodeBlock code={ragArchitectureExample} language="text" />
              </div>
              <p className="text-gray-600">
                The diagram above illustrates how user queries are processed through our RAG system
                to generate accurate, contextual responses using the available documentation.
              </p>
            </section>

            {/* API Reference */}
            <section className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">API Reference</h2>
              <p className="text-gray-600 mb-4">Available hooks, types, and their functionality:</p>
              <CodeBlock code={hookExample} />
            </section>
          </div>
        </div>
      </main>

      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-6xl h-[800px] flex flex-col">
            <div className="flex justify-between items-center p-4 border-b dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Chat with Aptos AI
              </h3>
              <button
                onClick={() => setIsOpen(false)}
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                Close
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              <ChatbotProvider
                config={{
                  apiKey: '',
                  apiUrl: 'http://localhost:8000',
                  ragProvider: RagProvider.DEVELOPER_DOCS,
                }}
              >
                <ChatWidgetContainer />
              </ChatbotProvider>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
