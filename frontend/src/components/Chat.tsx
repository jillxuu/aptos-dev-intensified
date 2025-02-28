import { useState, useRef, useEffect } from 'react'
import axios, { AxiosError } from 'axios'
import { PulseLoader } from 'react-spinners'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { motion, AnimatePresence } from 'framer-motion'
import { Toaster, toast } from 'react-hot-toast'
import { FiSend, FiCpu, FiThumbsUp, FiThumbsDown, FiCopy, FiCheck, FiTrash2 } from 'react-icons/fi'
import rainbowPet from '../assets/rainbow-pet-small.png'
import robotIcon from '../assets/robot-small.png'
import { config } from '../config'
import { v4 as uuidv4 } from 'uuid'

interface Message {
  id?: string
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
  feedback?: {
    rating?: boolean
    feedbackText?: string
    category?: string
  }
  usedChunks?: Array<{
    content: string
    section: string
    source: string
  }>
  timestamp?: string
}

interface ChatHistoryItem {
  id: string
  title: string
  timestamp: string
}

// Function to convert plain URLs to markdown links
const convertUrlsToMarkdown = (text: string): string => {
  // Match URLs that start with aptos.dev
  const aptosUrlRegex = /(aptos\.dev\/[^\s)]+)/g;
  return text.replace(aptosUrlRegex, '[$1](https://$1)');
}

const THEMES = [
  { name: 'cyberpunk', label: '🌆 Cyberpunk', icon: '🤖' },
  { name: 'synthwave', label: '🌅 Synthwave', icon: '🎵' },
  { name: 'retro', label: '🎮 Retro', icon: '👾' },
  { name: 'night', label: '🌙 Night', icon: '✨' }
]

// Modify the FEEDBACK_CATEGORIES constant to be an array instead of an object
const FEEDBACK_CATEGORIES = [
  {
    value: 'incorrect',
    label: 'Incorrect Information',
    description: 'The response contains factually incorrect information'
  },
  {
    value: 'incomplete',
    label: 'Incomplete Answer',
    description: 'The response is missing important information'
  },
  {
    value: 'unclear',
    label: 'Unclear Explanation',
    description: 'The response is confusing or poorly explained'
  },
  {
    value: 'not_helpful',
    label: 'Not Helpful',
    description: 'The response does not address my question'
  },
  {
    value: 'outdated',
    label: 'Outdated Information',
    description: 'The information appears to be outdated'
  },
  {
    value: 'other',
    label: 'Other',
    description: 'Other issues not listed above'
  }
]

const Chat = () => {
  const lastResponseRef = useRef<HTMLDivElement>(null)
  const loadingRef = useRef<HTMLDivElement>(null)
  const [theme, setTheme] = useState('cyberpunk')
  
  // State for chat
  const [chatId, setChatId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([{
    role: 'assistant',
    content: "👋 Hi there! I'm here to assist you with your queries about the Aptos blockchain technology. Feel free to ask me anything about:\n\n" +
      "- Move programming language\n" +
      "- Smart contracts development\n" +
      "- Account management\n" +
      "- Transactions and gas fees\n" +
      "- Network architecture\n" +
      "- Token standards\n" +
      "- And much more!\n\n" +
      "What would you like to learn about? 🚀"
  }])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null)
  const [clientId, setClientId] = useState<string>('')
  const [chatHistories, setChatHistories] = useState<ChatHistoryItem[]>([])
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [streamingMessage, setStreamingMessage] = useState<string>('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [feedbackModalOpen, setFeedbackModalOpen] = useState(false)
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null)
  const [selectedCategory, setSelectedCategory] = useState<string>('')
  const [feedbackText, setFeedbackText] = useState('')

  // Initialize client ID
  useEffect(() => {
    const storedClientId = localStorage.getItem('clientId')
    if (storedClientId) {
      setClientId(storedClientId)
    } else {
      const newClientId = uuidv4()
      localStorage.setItem('clientId', newClientId)
      setClientId(newClientId)
    }
  }, [])

  // Load chat histories when client ID is available
  useEffect(() => {
    const loadChatHistories = async () => {
      if (!clientId) return
      
      try {
        const response = await axios.get(`${config.apiBaseUrl}/chat/histories?client_id=${clientId}`)
        setChatHistories(response.data.histories)
      } catch (err) {
        console.error('Error loading chat histories:', err)
        toast.error('Failed to load chat histories')
      }
    }

    loadChatHistories()
  }, [clientId])

  // Load chat history when chatId is available
  useEffect(() => {
    const loadChatHistory = async () => {
      if (!chatId) return
      
      try {
        const response = await axios.get(`${config.apiBaseUrl}/chat/${chatId}/messages`)
        setMessages(response.data.messages)
      } catch (err) {
        console.error('Error loading chat history:', err)
        toast.error('Failed to load chat history')
        // If we can't load the history, reset to new chat
        setChatId(null)
        setMessages([{
          role: 'assistant',
          content: "👋 Hi there! I'm here to assist you with your queries about the Aptos blockchain technology..."
        }])
      }
    }

    loadChatHistory()
  }, [chatId])

  const scrollToLastResponse = () => {
    if (lastResponseRef.current) {
      lastResponseRef.current.scrollIntoView({ behavior: "smooth", block: "start" })
    }
  }

  // Scroll to loading indicator when it appears
  useEffect(() => {
    if (isLoading && loadingRef.current) {
      loadingRef.current.scrollIntoView({ behavior: "smooth", block: "end" })
    }
  }, [isLoading])

  const handleFeedback = async (messageId: string, rating: boolean, category?: string, feedbackText?: string) => {
    try {
      const message = messages.find(m => m.id === messageId)
      if (!message || message.role !== 'assistant') return

      const userMessage = messages[messages.findIndex(m => m.id === messageId) - 1]
      if (!userMessage || userMessage.role !== 'user') return

      // Update UI immediately
      setMessages(prev => prev.map(m => 
        m.id === messageId 
          ? { ...m, feedback: { rating, feedbackText, category } }
          : m
      ))

      // Send feedback to backend
      await axios.post(`${config.apiBaseUrl}/feedback`, {
        message_id: messageId,
        query: userMessage.content,
        response: message.content,
        rating,
        category,
        feedback_text: feedbackText,
        used_chunks: message.usedChunks,
        timestamp: new Date().toISOString()
      })

      toast.success('Thank you for your feedback!', {
        icon: rating ? '👍' : '👎',
        duration: 2000
      })
    } catch (err) {
      console.error('Error submitting feedback:', err)
      toast.error('Failed to submit feedback')
    }
  }

  const handleCopy = async (messageId: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content)
      setCopiedMessageId(messageId)
      toast.success('Copied to clipboard!', { duration: 2000 })
      // Reset the copied state after 2 seconds
      setTimeout(() => setCopiedMessageId(null), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
      toast.error('Failed to copy text')
    }
  }

  const handleStreamResponse = async (response: Response) => {
    const reader = response.body?.getReader()
    if (!reader) return

    setIsStreaming(true)
    let streamedContent = ''
    const streamingMessageId = uuidv4()

    try {
      // Add a temporary streaming message immediately
      setMessages(prev => [...prev, {
        id: streamingMessageId,
        role: 'assistant',
        content: ''
      }])

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        // Convert the chunk to text and append to the streaming message
        const chunk = new TextDecoder().decode(value)
        streamedContent += chunk
        
        // Update the last message with new content
        setMessages(prev => prev.map((msg, idx) => 
          idx === prev.length - 1 ? { ...msg, content: streamedContent } : msg
        ))
      }

      // After streaming is complete, save to database
      if (!chatId) {
        // Create new chat history
        const newChatId = uuidv4()
        const newChat = {
          id: newChatId,
          title: input.slice(0, 50) + "...",
          timestamp: new Date().toISOString(),
          messages: [
            {
              id: uuidv4(),
              role: 'user',
              content: input,
              timestamp: new Date().toISOString()
            },
            {
              id: streamingMessageId,
              role: 'assistant',
              content: streamedContent,
              timestamp: new Date().toISOString()
            }
          ],
          client_id: clientId
        }

        // Save new chat to database
        await axios.post(`${config.apiBaseUrl}/chat/history`, newChat)
        setChatId(newChatId)
        
        // Update chat histories list
        setChatHistories(prev => [{
          id: newChatId,
          title: input.slice(0, 50) + "...",
          timestamp: new Date().toISOString()
        }, ...prev])

      } else {
        // Update existing chat history
        const history = await axios.get(`${config.apiBaseUrl}/chat/${chatId}/messages`)
        const updatedHistory = {
          id: chatId,
          title: history.data.title,
          timestamp: new Date().toISOString(),
          client_id: clientId,
          messages: [
            ...history.data.messages,
            {
              id: streamingMessageId,
              role: 'assistant',
              content: streamedContent,
              timestamp: new Date().toISOString()
            }
          ]
        }
        await axios.put(`${config.apiBaseUrl}/chat/history/${chatId}`, updatedHistory)
      }

    } finally {
      reader.releaseLock()
      setIsStreaming(false)
    }
  }

  const handleSubmit = async () => {
    if (!input.trim() || !clientId) return

    const userMessage: Message = { 
      role: 'user', 
      content: input,
      id: uuidv4(),
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      let response;
      if (!chatId) {
        response = await fetch(`${config.apiBaseUrl}/chat/new/stream`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            messages: [userMessage],
            client_id: clientId
          })
        })
      } else {
        // First save the user message to the existing chat
        const history = await axios.get(`${config.apiBaseUrl}/chat/${chatId}/messages`)
        const updatedHistory = {
          id: chatId,
          title: history.data.title,
          timestamp: new Date().toISOString(),
          client_id: clientId,
          messages: [...history.data.messages, userMessage]
        }
        await axios.put(`${config.apiBaseUrl}/chat/history/${chatId}`, updatedHistory)

        response = await fetch(`${config.apiBaseUrl}/chat/${chatId}/message/stream`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            ...userMessage,
            client_id: clientId
          })
        })
      }

      if (!response.ok) {
        throw new Error('Failed to get response from the assistant')
      }

      await handleStreamResponse(response)

      toast.success('Response received!', {
        icon: '🤖',
        duration: 2000
      })
      setTimeout(scrollToLastResponse, 100)
    } catch (err) {
      toast.error('Failed to get response from the assistant')
      console.error('Chat error:', err)
      // Remove the user message if the request failed
      setMessages(prev => prev.slice(0, -1))
    } finally {
      setIsLoading(false)
    }
  }

  const startNewChat = () => {
    setChatId(null)
    setMessages([{
      role: 'assistant',
      content: "👋 Hi there! I'm here to assist you with your queries about the Aptos blockchain technology. Feel free to ask me anything about:\n\n" +
        "- Move programming language\n" +
        "- Smart contracts development\n" +
        "- Account management\n" +
        "- Transactions and gas fees\n" +
        "- Network architecture\n" +
        "- Token standards\n" +
        "- And much more!\n\n" +
        "What would you like to learn about? 🚀"
    }])
  }

  const handleDeleteChat = async (chatToDelete: ChatHistoryItem, e: React.MouseEvent) => {
    e.stopPropagation() // Prevent chat selection when clicking delete

    // Show confirmation dialog
    if (!window.confirm('Are you sure you want to delete this chat? This action cannot be undone.')) {
      return
    }

    try {
      await axios.delete(`${config.apiBaseUrl}/chat/history/${chatToDelete.id}`)
      
      // Remove from chat histories list
      setChatHistories(prev => prev.filter(chat => chat.id !== chatToDelete.id))
      
      // If the deleted chat was selected, reset to new chat
      if (chatId === chatToDelete.id) {
        startNewChat()
      }

      toast.success('Chat deleted successfully')
    } catch (err) {
      console.error('Error deleting chat:', err)
      toast.error('Failed to delete chat')
    }
  }

  const openFeedbackModal = (messageId: string) => {
    setSelectedMessageId(messageId)
    setSelectedCategory('')
    setFeedbackText('')
    setFeedbackModalOpen(true)
  }

  const submitNegativeFeedback = () => {
    if (!selectedMessageId) return
    handleFeedback(selectedMessageId, false, selectedCategory, feedbackText)
    setFeedbackModalOpen(false)
    setSelectedMessageId(null)
    setSelectedCategory('')
    setFeedbackText('')
  }

  return (
    <div className="min-h-screen" data-theme={theme}>
      {/* Chat History Sidebar */}
      <div className={`fixed left-0 top-0 h-full bg-base-200 transition-all duration-300 ${isSidebarOpen ? 'w-64' : 'w-0'} overflow-hidden flex flex-col`}>
        <div className="p-4 w-64">
          <div className="flex items-center gap-2 mb-6">
            <button
              className="btn btn-circle btn-sm flex-shrink-0"
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            >
              ←
            </button>
            <h2 className="text-xl font-bold truncate whitespace-nowrap">
              Chat History
            </h2>
          </div>

          {/* Scrollable Chat History */}
          <div className="space-y-2 overflow-y-auto" style={{ maxHeight: 'calc(100vh - 180px)' }}>
            {chatHistories.map((chat) => (
              <div
                key={chat.id}
                className={`p-2 rounded cursor-pointer hover:bg-base-300 ${chatId === chat.id ? 'bg-primary text-primary-content' : ''} flex justify-between items-center group`}
                onClick={() => {
                  setChatId(chat.id)
                }}
              >
                <div className="flex-1 min-w-0">
                  <div className="font-medium truncate">{chat.title}</div>
                  <div className="text-xs opacity-70">
                    {new Date(chat.timestamp).toLocaleDateString()}
                  </div>
                </div>
                <button
                  className="btn btn-ghost btn-xs opacity-0 group-hover:opacity-100 transition-opacity"
                  onClick={(e) => handleDeleteChat(chat, e)}
                  title="Delete chat"
                >
                  <FiTrash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Fixed Start New Chat Button at Bottom */}
        <div className="p-4 w-64 mt-auto border-t border-base-300">
          <button
            className="btn btn-primary w-full"
            onClick={startNewChat}
          >
            Start New Chat
          </button>
        </div>
      </div>

      {/* Toggle Sidebar Button (only shown when sidebar is closed) */}
      {!isSidebarOpen && (
        <button
          className="fixed left-4 top-4 btn btn-circle btn-sm"
          onClick={() => setIsSidebarOpen(true)}
        >
          →
        </button>
      )}

      {/* Theme Selector */}
      <div className="fixed top-4 right-4 dropdown dropdown-end z-50">
        <div tabIndex={0} role="button" className="btn btn-primary btn-sm">
          <span>{THEMES.find(t => t.name === theme)?.icon}</span>
          <span>Theme</span>
        </div>
        <ul tabIndex={0} className="dropdown-content z-[1] menu p-2 shadow-xl bg-base-200 rounded-box w-52">
          {THEMES.map((t) => (
            <li key={t.name}>
              <a 
                className={`flex items-center gap-2 ${theme === t.name ? 'active' : ''}`}
                onClick={() => setTheme(t.name)}
              >
                <span>{t.icon}</span>
                <span>{t.label}</span>
              </a>
            </li>
          ))}
        </ul>
      </div>

      <div className={`container mx-auto p-4 min-h-screen ${isSidebarOpen ? 'ml-64' : ''}`}>
        <div className="card min-h-screen">
          {/* Header */}
          <div className="card-body items-center text-center pb-2">
            <h1 className="card-title text-4xl font-bold mb-2">
              Sudo Make Me Smart 🧠
            </h1>
            <p className="opacity-70">
              Your friendly neighborhood blockchain expert
            </p>
            {chatId && (
              <button
                className="btn btn-sm btn-ghost mt-2"
                onClick={startNewChat}
              >
                Start New Chat
              </button>
            )}
          </div>

          {/* Messages */}
          <div className="card-body py-4 gap-4 overflow-y-auto max-h-[calc(100vh-16rem)] scroll-smooth">
            <AnimatePresence>
              {messages.map((message, index) => (
                <motion.div
                  key={index}
                  ref={index === messages.length - 1 && message.role === 'assistant' ? lastResponseRef : null}
                  className="chat chat-start"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                >
                  <div className="chat-image avatar">
                    <div className="w-10 mask mask-squircle bg-base-200 p-1">
                      <img src={message.role === 'assistant' ? robotIcon : rainbowPet} alt={message.role} />
                    </div>
                  </div>
                  <div className={`chat-bubble ${message.role === 'assistant' ? 'chat-bubble-primary bg-primary/90' : 'chat-bubble bg-base-200'}`}>
                    <div className="prose prose-sm max-w-none">
                      <ReactMarkdown 
                        remarkPlugins={[remarkGfm]}
                        components={{
                          a: (props) => (
                            <a 
                              {...props} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className={`link font-semibold ${message.role === 'assistant' ? 'text-primary-content' : 'text-base-content'} underline decoration-2 opacity-90 hover:opacity-100 transition-opacity`}
                            />
                          ),
                          p: (props) => (
                            <p {...props} className="mb-3 last:mb-0" />
                          ),
                          ul: (props) => (
                            <ul {...props} className="mb-3 list-disc pl-4" />
                          ),
                          ol: (props) => (
                            <ol {...props} className="mb-3 list-decimal pl-4" />
                          ),
                          li: (props) => (
                            <li {...props} className="mb-1" />
                          ),
                          code: (props) => (
                            <code {...props} className={`rounded px-1 py-0.5 ${message.role === 'assistant' ? 'bg-primary-focus/30' : 'bg-base-300/50'}`} />
                          ),
                          pre: (props) => (
                            <pre {...props} className={`rounded-lg p-3 my-2 overflow-x-auto ${message.role === 'assistant' ? 'bg-primary-focus/30' : 'bg-base-300/50'}`} />
                          ),
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
                      {message.role === 'assistant' && index === messages.length - 1 && isStreaming && (
                        <span className="inline-block animate-pulse">▊</span>
                      )}
                    </div>
                    {message.sources && message.sources.length > 0 && (
                      <>
                        <div className="divider">Sources</div>
                        <ul className="menu menu-xs bg-primary-focus/30 rounded-box p-2">
                          {message.sources.map((source, idx) => (
                            <li key={idx}>
                              <a
                                href={`https://${source}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="link text-primary-content hover:text-primary-content/80 font-medium"
                              >
                                {source}
                              </a>
                            </li>
                          ))}
                        </ul>
                      </>
                    )}
                    {message.role === 'assistant' && message.id && (
                      <div className="flex items-center gap-2 mt-2">
                        <button
                          className={`btn btn-sm btn-ghost text-primary-content ${message.feedback?.rating === true ? 'btn-success' : ''}`}
                          onClick={() => handleFeedback(message.id!, true)}
                          disabled={message.feedback !== undefined}
                        >
                          <FiThumbsUp />
                        </button>
                        <button
                          className={`btn btn-sm btn-ghost text-primary-content ${message.feedback?.rating === false ? 'btn-error' : ''}`}
                          onClick={() => openFeedbackModal(message.id!)}
                          disabled={message.feedback !== undefined}
                        >
                          <FiThumbsDown />
                        </button>
                        <button
                          className="btn btn-sm btn-ghost text-primary-content"
                          onClick={() => handleCopy(message.id!, message.content)}
                          title="Copy response"
                        >
                          {copiedMessageId === message.id ? <FiCheck className="text-success" /> : <FiCopy />}
                        </button>
                      </div>
                    )}
                  </div>
                </motion.div>
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

          {/* Input */}
          <div className="card-body pt-2">
            <div className="join w-full">
              <input
                type="text"
                placeholder="Ask me anything about Aptos..."
                className="input input-bordered join-item flex-1"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
                disabled={isLoading}
              />
              <button
                className="btn btn-primary join-item"
                onClick={handleSubmit}
                disabled={isLoading}
              >
                {isLoading ? (
                  <PulseLoader size={4} />
                ) : (
                  <>
                    Send
                    <FiSend />
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Add Feedback Modal */}
      {feedbackModalOpen && (
        <div className="modal modal-open">
          <div className="modal-box">
            <h3 className="font-bold text-lg mb-4">What could be improved?</h3>
            
            <div className="form-control w-full">
              <label className="label">
                <span className="label-text">Category</span>
              </label>
              <select 
                className="select select-bordered w-full" 
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                required
              >
                <option value="">Select a category</option>
                {FEEDBACK_CATEGORIES.map((cat) => (
                  <option key={cat.value} value={cat.value}>
                    {cat.label}
                  </option>
                ))}
              </select>
              {selectedCategory && (
                <label className="label">
                  <span className="label-text-alt">
                    {FEEDBACK_CATEGORIES.find(cat => cat.value === selectedCategory)?.description}
                  </span>
                </label>
              )}
            </div>

            <div className="form-control w-full mt-4">
              <label className="label">
                <span className="label-text">Additional Comments</span>
              </label>
              <textarea 
                className="textarea textarea-bordered h-24"
                placeholder="Please provide more details about the issue..."
                value={feedbackText}
                onChange={(e) => setFeedbackText(e.target.value)}
              />
            </div>

            <div className="modal-action">
              <button 
                className="btn btn-error" 
                onClick={submitNegativeFeedback}
                disabled={!selectedCategory}
              >
                Submit Feedback
              </button>
              <button 
                className="btn btn-ghost" 
                onClick={() => setFeedbackModalOpen(false)}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <Toaster position="bottom-right" />
    </div>
  )
}

export default Chat