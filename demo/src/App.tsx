import { useState } from 'react';
import { ChatbotProvider, useChatbot } from '@aptos/chatbot-react';
import { ChatWidget } from '@aptos/chatbot-ui-base';
import { RagProvider } from '@aptos/chatbot-core';

function ChatWidgetContainer() {
  const {
    messages = [],
    sendMessage,
    isLoading = false,
    isGenerating = false,
    isTyping = false,
    hasMoreMessages = false,
    fastMode = false,
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
    setFastMode,
  } = useChatbot();

  return (
    <ChatWidget
      messages={messages}
      onSendMessage={sendMessage}
      isLoading={isLoading}
      isGenerating={isGenerating}
      isTyping={isTyping}
      hasMoreMessages={hasMoreMessages}
      fastMode={fastMode}
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
      onToggleFastMode={setFastMode}
      className="h-full"
    />
  );
}

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
                  apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:8080',
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
