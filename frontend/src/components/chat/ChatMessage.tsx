import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { motion } from "framer-motion";
import {
  FiThumbsUp,
  FiThumbsDown,
  FiCopy,
  FiCheck,
  FiInfo,
  FiX,
} from "react-icons/fi";
import rainbowPet from "../../assets/rainbow-pet-small.png";
import robotIcon from "../../assets/robot-small.png";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Message } from "./types";

interface MessageProps {
  message: Message;
  isLastMessage: boolean;
  isStreaming: boolean;
  copiedMessageId: string | null;
  onFeedback: (messageId: string, rating: boolean) => void;
  onOpenFeedbackModal: (messageId: string) => void;
  onCopy: (messageId: string, content: string) => void;
  forwardedRef?: React.RefObject<HTMLDivElement>;
}

const ChatMessage: React.FC<MessageProps> = ({
  message,
  isLastMessage,
  isStreaming,
  copiedMessageId,
  onFeedback,
  onOpenFeedbackModal,
  onCopy,
  forwardedRef,
}) => {
  const [showMetadata, setShowMetadata] = useState(false);

  // Format metadata for display
  const formatMetadata = (metadata: any) => {
    if (!metadata) return "No metadata available";

    // Format the metadata as JSON with indentation for readability
    return JSON.stringify(metadata, null, 2);
  };

  return (
    <motion.div
      ref={forwardedRef}
      className="mb-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
    >
      {/* First row: Icon */}
      <div className="flex mb-1">
        <div className="w-10 mask mask-squircle bg-base-200 p-1">
          <img
            src={message.role === "assistant" ? robotIcon : rainbowPet}
            alt={message.role}
          />
        </div>
      </div>

      {/* Second row: Message content */}
      <div className="ml-2">
        <div className="prose prose-sm max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: (props) => (
                <a
                  {...props}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={
                    "link font-semibold underline decoration-2 opacity-90 hover:opacity-100 transition-opacity"
                  }
                />
              ),
              p: (props) => <p {...props} className="mb-3 last:mb-0" />,
              ul: (props) => <ul {...props} className="mb-3 list-disc pl-4" />,
              ol: (props) => (
                <ol {...props} className="mb-3 list-decimal pl-4" />
              ),
              li: (props) => <li {...props} className="mb-1" />,
              code: (props: any) => {
                const { children, className, ...rest } = props;
                const match = /language-(\w+)/.exec(className || "");
                const language = match ? match[1] : "";
                const inline = !className;

                return !inline ? (
                  <SyntaxHighlighter
                    style={vscDarkPlus}
                    language={language || "text"}
                    PreTag="div"
                    className="rounded-lg my-2 overflow-x-auto"
                    wrapLines={true}
                    {...rest}
                  >
                    {String(children).replace(/\n$/, "")}
                  </SyntaxHighlighter>
                ) : (
                  <code
                    className={`rounded px-1 py-0.5 ${
                      message.role === "assistant"
                        ? "bg-primary-focus/30"
                        : "bg-base-300/50"
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
          {message.role === "assistant" && isLastMessage && isStreaming && (
            <span className="inline-block animate-pulse">▊</span>
          )}
        </div>

        {/* Metadata display */}
        {showMetadata && message.metadata && (
          <div className="mt-2 p-3 bg-base-200 rounded-lg relative">
            <button
              className="absolute top-2 right-2 btn btn-xs btn-circle"
              onClick={() => setShowMetadata(false)}
            >
              <FiX />
            </button>
            <h4 className="text-sm font-semibold mb-2">Message Metadata</h4>
            <pre className="text-xs overflow-x-auto whitespace-pre-wrap">
              {formatMetadata(message.metadata)}
            </pre>
          </div>
        )}

        {message.role === "assistant" && message.id && (
          <div className="flex items-center gap-2 mt-2">
            <button
              className={`btn btn-sm btn-ghost ${
                message.feedback?.rating === true ? "btn-success" : ""
              }`}
              onClick={() => onFeedback(message.id!, true)}
              disabled={message.feedback !== undefined}
            >
              <FiThumbsUp />
            </button>
            <button
              className={`btn btn-sm btn-ghost ${
                message.feedback?.rating === false ? "btn-error" : ""
              }`}
              onClick={() => onOpenFeedbackModal(message.id!)}
              disabled={message.feedback !== undefined}
            >
              <FiThumbsDown />
            </button>
            <button
              className="btn btn-sm btn-ghost"
              onClick={() => onCopy(message.id!, message.content)}
              title="Copy response"
            >
              {copiedMessageId === message.id ? (
                <FiCheck className="text-success" />
              ) : (
                <FiCopy />
              )}
            </button>
            {message.metadata && (
              <button
                className="btn btn-sm btn-ghost"
                onClick={() => setShowMetadata(!showMetadata)}
                title={showMetadata ? "Hide metadata" : "View metadata"}
              >
                <FiInfo className={showMetadata ? "text-primary" : ""} />
              </button>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default React.memo(ChatMessage);
