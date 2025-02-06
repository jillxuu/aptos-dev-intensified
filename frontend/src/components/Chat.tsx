import { useState, useRef, useEffect } from 'react'
import axios, { AxiosError } from 'axios'
import { PulseLoader } from 'react-spinners'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { motion, AnimatePresence } from 'framer-motion'
import { Toaster, toast } from 'react-hot-toast'
import { FiSend, FiCpu, FiThumbsUp, FiThumbsDown, FiCopy, FiCheck } from 'react-icons/fi'
import rainbowPet from '../assets/rainbow-pet-small.png'
import robotIcon from '../assets/robot-small.png'
import { config } from '../config'

interface Message {
  id?: string
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
  feedback?: {
    rating?: boolean
    feedbackText?: string
  }
  usedChunks?: Array<{
    content: string
    section: string
    source: string
  }>
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

const Chat = () => {
  const lastResponseRef = useRef<HTMLDivElement>(null)
  const loadingRef = useRef<HTMLDivElement>(null)
  const [theme, setTheme] = useState('cyberpunk')
  const [messages, setMessages] = useState<Message[]>([
    {
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
    }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null)

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

  const handleFeedback = async (messageId: string, rating: boolean, feedbackText?: string) => {
    try {
      const message = messages.find(m => m.id === messageId)
      if (!message || message.role !== 'assistant') return

      const userMessage = messages[messages.findIndex(m => m.id === messageId) - 1]
      if (!userMessage || userMessage.role !== 'user') return

      // Update UI immediately
      setMessages(prev => prev.map(m => 
        m.id === messageId 
          ? { ...m, feedback: { rating, feedbackText } }
          : m
      ))

      // Send feedback to backend
      await axios.post(`${config.apiBaseUrl}/feedback`, {
        message_id: messageId,
        query: userMessage.content,
        response: message.content,
        rating,
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

  const handleSubmit = async () => {
    if (!input.trim()) return

    const userMessage: Message = { 
      id: crypto.randomUUID(),
      role: 'user', 
      content: input 
    }

    // Create new messages array
    const newMessages = [...messages, userMessage]
    setMessages(newMessages)
    setInput('')
    setIsLoading(true)

    try {
      const response = await axios.post(`${config.apiBaseUrl}/chat`, {
        messages: newMessages,
        temperature: 0.7
      })

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: convertUrlsToMarkdown(response.data.response),
        sources: response.data.sources,
        usedChunks: response.data.used_chunks
      }

      // Update messages with assistant response
      const updatedMessages = [...newMessages, assistantMessage]
      setMessages(updatedMessages)

      toast.success('Response received!', {
        icon: '🤖',
        duration: 2000
      })
      setTimeout(scrollToLastResponse, 100)
    } catch (err) {
      const error = err as AxiosError
      toast.error('Failed to get response from the assistant')
      console.error('Chat error:', error.message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen" data-theme={theme}>
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

      <div className="container mx-auto p-4 min-h-screen">
        <div className="card min-h-screen">
          {/* Header */}
          <div className="card-body items-center text-center pb-2">
            <h1 className="card-title text-4xl font-bold mb-2">
              Sudo Make Me Smart 🧠
            </h1>
            <p className="opacity-70">
              Your friendly neighborhood blockchain expert
            </p>
          </div>

          {/* Messages */}
          <div className="card-body py-4 gap-4 overflow-y-auto max-h-[calc(100vh-16rem)] scroll-smooth">
            <AnimatePresence>
              {messages.map((message, index) => (
                <motion.div
                  key={index}
                  ref={index === messages.length - 1 && message.role === 'assistant' ? lastResponseRef : null}
                  className={`chat ${message.role === 'assistant' ? 'chat-start' : 'chat-end'}`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                >
                  <div className="chat-image avatar">
                    <div className={`w-10 mask ${message.role === 'assistant' ? 'mask-squircle' : 'mask-circle'}`}>
                      <img src={message.role === 'assistant' ? robotIcon : rainbowPet} alt={message.role} />
                    </div>
                  </div>
                  <div className={`chat-bubble ${message.role === 'assistant' ? 'chat-bubble-primary' : 'chat-bubble-secondary'}`}>
                    <div className="prose prose-sm max-w-none">
                      <ReactMarkdown 
                        remarkPlugins={[remarkGfm]}
                        components={{
                          a: (props) => (
                            <a 
                              {...props} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="link font-semibold text-info hover:text-info-focus underline decoration-2 opacity-90 hover:opacity-100 transition-opacity"
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
                            <code {...props} className="bg-base-200/50 rounded px-1 py-0.5" />
                          ),
                          pre: (props) => (
                            <pre {...props} className="bg-base-200/50 rounded-lg p-3 my-2 overflow-x-auto" />
                          ),
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
                    </div>
                    {message.sources && message.sources.length > 0 && (
                      <>
                        <div className="divider">Sources</div>
                        <ul className="menu menu-xs bg-base-200/30 rounded-box p-2">
                          {message.sources.map((source, idx) => (
                            <li key={idx}>
                              <a
                                href={`https://${source}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="link text-info hover:text-info-focus font-medium"
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
                          className={`btn btn-sm btn-ghost ${message.feedback?.rating === true ? 'btn-success' : ''}`}
                          onClick={() => handleFeedback(message.id!, true)}
                          disabled={message.feedback !== undefined}
                        >
                          <FiThumbsUp />
                        </button>
                        <button
                          className={`btn btn-sm btn-ghost ${message.feedback?.rating === false ? 'btn-error' : ''}`}
                          onClick={() => {
                            const feedbackText = prompt('What could be improved?')
                            if (feedbackText !== null) {
                              handleFeedback(message.id!, false, feedbackText)
                            }
                          }}
                          disabled={message.feedback !== undefined}
                        >
                          <FiThumbsDown />
                        </button>
                        <button
                          className="btn btn-sm btn-ghost"
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
      <Toaster position="bottom-right" />
    </div>
  )
}

export default Chat