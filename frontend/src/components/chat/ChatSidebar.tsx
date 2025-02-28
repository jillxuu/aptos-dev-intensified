import React from "react";
import { FiTrash2 } from "react-icons/fi";

interface ChatHistoryItem {
  id: string;
  title: string;
  timestamp: string;
}

interface ChatSidebarProps {
  isSidebarOpen: boolean;
  chatHistories: ChatHistoryItem[];
  currentChatId: string | null;
  onToggleSidebar: () => void;
  onSelectChat: (chatId: string) => void;
  onDeleteChat: (chat: ChatHistoryItem, e: React.MouseEvent) => void;
  onStartNewChat: () => void;
}

const ChatSidebar: React.FC<ChatSidebarProps> = ({
  isSidebarOpen,
  chatHistories,
  onToggleSidebar,
  onSelectChat,
  onDeleteChat,
  onStartNewChat,
}) => {
  return (
    <>
      <div
        className={`absolute left-0 top-0 h-full bg-base-200 transition-all duration-300 ${
          isSidebarOpen ? "w-64" : "w-0"
        } overflow-hidden flex flex-col z-20`}
        style={{ maxHeight: "100%" }}
      >
        <div className="p-4 w-64 flex flex-col flex-grow overflow-hidden">
          <div className="flex items-center gap-2 mb-6">
            <button
              className="btn btn-circle btn-sm flex-shrink-0"
              onClick={onToggleSidebar}
            >
              ←
            </button>
            <h2 className="text-xl font-bold truncate whitespace-nowrap">
              Chat History
            </h2>
          </div>

          {/* Scrollable Chat History */}
          <div className="space-y-2 overflow-y-auto flex-grow">
            {chatHistories.map((chat) => (
              <div
                key={chat.id}
                className={`p-2 rounded cursor-pointer hover:bg-base-300 flex justify-between items-center group`}
                onClick={() => onSelectChat(chat.id)}
              >
                <div className="flex-1 min-w-0">
                  <div className="font-medium truncate">{chat.title}</div>
                  <div className="text-xs opacity-70">
                    {new Date(chat.timestamp).toLocaleDateString()}
                  </div>
                </div>
                <button
                  className="btn btn-ghost btn-xs opacity-0 group-hover:opacity-100 transition-opacity"
                  onClick={(e) => onDeleteChat(chat, e)}
                  title="Delete chat"
                >
                  <FiTrash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>

          {/* Start New Chat Button at Bottom */}
          <div className="mt-4 pt-2 border-t border-base-300">
            <button className="btn btn-primary w-full" onClick={onStartNewChat}>
              Start New Chat
            </button>
          </div>
        </div>
      </div>

      {/* Toggle Sidebar Button (only shown when sidebar is closed) */}
      {!isSidebarOpen && (
        <button
          className={"absolute left-4 top-4 btn btn-circle btn-sm z-20"}
          onClick={onToggleSidebar}
        >
          →
        </button>
      )}
    </>
  );
};

export default React.memo(ChatSidebar);
