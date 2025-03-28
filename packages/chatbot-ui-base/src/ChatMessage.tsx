import { type ReactElement } from 'react';
import { type ChatMessageProps } from '@aptos/chatbot-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter, SyntaxHighlighterProps } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './ChatMessage.css';

const PrismSyntaxHighlighter = SyntaxHighlighter as unknown as React.FC<SyntaxHighlighterProps>;

export function ChatMessage({
  message,
  detailed = false,
  onCopy,
  onFeedback,
  className = '',
}: ChatMessageProps): ReactElement {
  const handleCopy = () => {
    onCopy?.(message.id);
  };

  const handleFeedback = (feedback: 'positive' | 'negative') => {
    onFeedback?.(message.id, feedback);
  };

  return (
    <div className={`chat-message ${message.role} ${className}`}>
      <div className="message-content">
        <div className="prose prose-sm max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: props => (
                <a
                  {...props}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="link font-semibold underline decoration-2 opacity-90 hover:opacity-100 transition-opacity"
                />
              ),
              p: props => <p {...props} className="mb-3 last:mb-0" />,
              ul: props => <ul {...props} className="mb-3 list-disc pl-4" />,
              ol: props => <ol {...props} className="mb-3 list-decimal pl-4" />,
              li: props => <li {...props} className="mb-1" />,
              code: (props: any) => {
                const { children, className, ...rest } = props;
                const match = /language-(\w+)/.exec(className || '');
                const language = match ? match[1] : '';
                const inline = !className;

                return !inline ? (
                  <PrismSyntaxHighlighter
                    style={vscDarkPlus}
                    language={language || 'text'}
                    PreTag="div"
                    className="rounded-lg my-2 overflow-x-auto"
                    wrapLines={true}
                    {...rest}
                  >
                    {String(children).replace(/\n$/, '')}
                  </PrismSyntaxHighlighter>
                ) : (
                  <code
                    className={`rounded px-1 py-0.5 ${
                      message.role === 'assistant' ? 'bg-primary-focus/30' : 'bg-base-300/50'
                    }`}
                    {...rest}
                  >
                    {children}
                  </code>
                );
              },
              pre: ({ children }) => <>{children}</>,
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>
        {detailed && (
          <div className="message-details">
            <span className="message-timestamp">
              {new Date(message.timestamp).toLocaleString()}
            </span>
            <span className="message-id">ID: {message.id}</span>
          </div>
        )}
      </div>
      <div className="message-actions">
        <button onClick={handleCopy} className="copy-button">
          Copy
        </button>
        {message.role === 'assistant' && (
          <div className="feedback-buttons">
            <button
              onClick={() => handleFeedback('positive')}
              className={`feedback-button ${message.feedback === 'positive' ? 'active' : ''}`}
            >
              👍
            </button>
            <button
              onClick={() => handleFeedback('negative')}
              className={`feedback-button ${message.feedback === 'negative' ? 'active' : ''}`}
            >
              👎
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
