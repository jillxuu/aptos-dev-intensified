import React, { useRef, useState } from "react";
import { PulseLoader } from "react-spinners";
import { FiSend } from "react-icons/fi";

interface ChatInputProps {
  isLoading: boolean;
  onSubmit: (message: string) => void;
}

const ChatInput: React.FC<ChatInputProps> = ({ isLoading, onSubmit }) => {
  const inputRef = useRef<HTMLInputElement>(null);
  // Manage input state completely internally
  const [localInput, setLocalInput] = useState("");

  // Handle input changes locally only
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setLocalInput(e.target.value);
  };

  const handleSubmit = () => {
    if (!localInput.trim() || isLoading) return;

    // Only communicate with parent when submitting
    onSubmit(localInput);
    // Clear the input field after submission
    setLocalInput("");
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !isLoading && localInput.trim()) {
      handleSubmit();
    }
  };

  return (
    <div className="flex-shrink-0 mb-1">
      <div className="join w-full">
        <input
          ref={inputRef}
          type="text"
          placeholder="Ask me anything about Aptos..."
          className="input input-bordered join-item flex-1"
          value={localInput}
          onChange={handleInputChange}
          onKeyDown={handleKeyPress}
          disabled={isLoading}
        />
        <button
          className="btn btn-primary join-item"
          onClick={handleSubmit}
          disabled={isLoading || !localInput.trim()}
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
  );
};

// Use React.memo to prevent unnecessary re-renders
export default React.memo(ChatInput);
