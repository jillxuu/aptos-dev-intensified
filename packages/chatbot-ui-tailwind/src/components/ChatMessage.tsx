import { type ReactElement } from 'react';
import { ChatMessage as BaseChatMessage } from '@aptos/chatbot-ui-base';
import { ChatMessageProps } from '@aptos/chatbot-react';

// Simple SVG components to replace Heroicons
const ClipboardIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.5}
    stroke="currentColor"
    className={className}
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184"
    />
  </svg>
);

const HandThumbUpIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.5}
    stroke="currentColor"
    className={className}
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M6.633 10.5c.806 0 1.533-.446 2.031-1.08a9.041 9.041 0 012.861-2.4c.723-.384 1.35-.956 1.653-1.715a4.498 4.498 0 00.322-1.672V3a.75.75 0 01.75-.75A2.25 2.25 0 0116.5 4.5c0 1.152-.26 2.243-.723 3.218-.266.558.107 1.282.725 1.282h3.126c1.026 0 1.945.694 2.054 1.715.045.422.068.85.068 1.285a11.95 11.95 0 01-2.649 7.521c-.388.482-.987.729-1.605.729H13.48c-.483 0-.964-.078-1.423-.23l-3.114-1.04a4.501 4.501 0 00-1.423-.23H5.904M14.25 9h2.25M5.904 18.75c.083.205.173.405.27.602.197.4-.078.898-.523.898h-.908c-.889 0-1.713-.518-1.972-1.368a12 12 0 01-.521-3.507c0-1.553.295-3.036.831-4.398C3.387 10.203 4.167 9.75 5 9.75h1.053c.472 0 .745.556.5.96a8.958 8.958 0 00-1.302 4.665c0 1.194.232 2.333.654 3.375z"
    />
  </svg>
);

const HandThumbDownIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.5}
    stroke="currentColor"
    className={className}
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M7.5 15h2.25m8.024-9.75c.011.05.028.1.052.148.591 1.2.924 2.55.924 3.977a8.96 8.96 0 01-.999 4.125m.023-8.25c-.076-.365.183-.75.575-.75h.908c.889 0 1.713.518 1.972 1.368.339 1.11.521 2.287.521 3.507 0 1.553-.295 3.036-.831 4.398C20.613 14.547 19.833 15 19 15h-1.053c-.472 0-.745-.556-.5-.96a8.95 8.95 0 00.303-.54m.023-8.25H16.48a4.5 4.5 0 01-1.423-.23l-3.114-1.04a4.5 4.5 0 00-1.423-.23H6.504c-.618 0-1.217.247-1.605.729A11.95 11.95 0 002.25 12c0 .434.023.863.068 1.285C2.427 14.306 3.346 15 4.372 15h3.126c.618 0 .991.724.725 1.282A7.471 7.471 0 007.5 19.5a2.25 2.25 0 002.25 2.25.75.75 0 00.75-.75v-.633c0-.573.11-1.14.322-1.672.304-.76.93-1.33 1.653-1.715a9.04 9.04 0 002.86-2.4c.498-.634 1.226-1.08 2.032-1.08h.384"
    />
  </svg>
);

export function ChatMessage({
  message,
  onCopy,
  onFeedback,
  detailed,
}: ChatMessageProps): ReactElement {
  const handleCopy = () => {
    onCopy?.(message.id);
  };

  const handleFeedback = (feedback: 'positive' | 'negative') => {
    onFeedback?.(message.id, feedback);
  };

  const messageClassName = `flex flex-col gap-2 p-4 rounded-lg ${
    message.role === 'assistant' ? 'bg-blue-50' : 'bg-gray-50'
  }`;

  const avatarClassName = `flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
    message.role === 'assistant'
      ? 'bg-blue-100 text-blue-600 dark:bg-blue-900 dark:text-blue-300'
      : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300'
  }`;

  const contentClassName = 'prose dark:prose-invert max-w-none';
  const actionsClassName = 'flex items-center gap-2';
  const copyButtonClassName =
    'inline-flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200';

  const getFeedbackButtonClassName = (isActive: boolean, type: 'positive' | 'negative') => {
    if (type === 'positive' && isActive) {
      return 'bg-green-100 text-green-600 dark:bg-green-900 dark:text-green-300';
    }
    if (type === 'negative' && isActive) {
      return 'bg-red-100 text-red-600 dark:bg-red-900 dark:text-red-300';
    }
    return 'text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300';
  };

  return (
    <div className={messageClassName}>
      <div className="flex items-start gap-4">
        <div className={avatarClassName}>{message.role === 'assistant' ? '🤖' : '👤'}</div>
        <div className="flex-1 space-y-2">
          <div className={contentClassName}>
            <BaseChatMessage
              message={message}
              onCopy={onCopy}
              onFeedback={onFeedback}
              detailed={detailed}
            />
          </div>
          <div className={actionsClassName}>
            <button onClick={handleCopy} className={copyButtonClassName}>
              <ClipboardIcon className="w-4 h-4" aria-hidden="true" />
              Copy
            </button>
            {message.role === 'assistant' && (
              <div className="flex items-center gap-1">
                <button
                  onClick={() => handleFeedback('positive')}
                  className={`p-1 rounded-full transition-colors ${getFeedbackButtonClassName(message.feedback === 'positive', 'positive')}`}
                >
                  <HandThumbUpIcon className="w-4 h-4" aria-hidden="true" />
                </button>
                <button
                  onClick={() => handleFeedback('negative')}
                  className={`p-1 rounded-full transition-colors ${getFeedbackButtonClassName(message.feedback === 'negative', 'negative')}`}
                >
                  <HandThumbDownIcon className="w-4 h-4" aria-hidden="true" />
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
