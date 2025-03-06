import { useState, useEffect } from "react";
import { Modal } from "./Modal";
import { TriggerButton } from "./TriggerButton";
import Chat from "../chat/Chat";
import { config as defaultConfig, Config } from "../../config";
import axios from "axios";

interface AptosChatbotPluginProps {
  className?: string;
  buttonText?: string;
  buttonClassName?: string;
  modalTitle?: string;
  apiUrl?: string;
  ragProvider?: string;
  ragConfig?: Record<string, any>;
  githubRepoUrl?: string;
  githubRepoBranch?: string;
}

/**
 * AptosChatbotPlugin - A standalone plugin component that can be embedded anywhere
 *
 * @param className - Additional class names for the container
 * @param buttonText - Custom text for the trigger button
 * @param buttonClassName - Additional class names for the button
 * @param modalTitle - Custom title for the modal header
 * @param apiUrl - Optional custom API URL to override the default backend URL
 * @param ragProvider - Optional custom RAG provider name to use
 * @param ragConfig - Optional configuration for the RAG provider
 * @param githubRepoUrl - GitHub repository URL to use as knowledge base (when ragProvider="github")
 * @param githubRepoBranch - Branch to use for the GitHub repository (defaults to "main")
 */
export const AptosChatbotPlugin: React.FC<AptosChatbotPluginProps> = ({
  className = "",
  buttonText = "Ask Aptos AI",
  buttonClassName = "",
  modalTitle = "Ask Aptos AI",
  apiUrl,
  ragProvider,
  ragConfig,
  githubRepoUrl,
  githubRepoBranch = "main",
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);
  const [initAttempts, setInitAttempts] = useState(0);
  const [isInitialized, setIsInitialized] = useState(false);

  // Create a custom config if any overrides are provided
  const customConfig: Config = {
    ...defaultConfig,
    ...(apiUrl && { apiBaseUrl: apiUrl }),
    ...(ragProvider && { ragProvider }),
    ...(ragConfig && { ragConfig }),
  };

  // Initialize GitHub provider if needed
  useEffect(() => {
    const initializeGitHubProvider = async () => {
      // Only initialize if ragProvider is github and githubRepoUrl is provided
      // and we haven't already initialized or tried too many times
      if (
        ragProvider === "github" &&
        githubRepoUrl &&
        !isInitialized &&
        initAttempts < 3
      ) {
        setIsInitializing(true);
        setInitError(null);

        try {
          await axios.post(
            `${customConfig.apiBaseUrl}/rag/provider/github/initialize`,
            {
              repo_url: githubRepoUrl,
              branch: githubRepoBranch,
            },
          );
          console.log("GitHub repository initialized successfully!");
          setIsInitialized(true);
        } catch (err: any) {
          console.error("Error initializing GitHub Knowledge provider:", err);

          // Extract the most helpful error message
          let errorMessage = "Failed to initialize GitHub Knowledge provider";

          if (err.response?.data?.detail) {
            errorMessage = err.response.data.detail;
          } else if (err.message) {
            errorMessage = err.message;
          }

          // Add more helpful context based on common errors
          if (errorMessage.includes("Failed to clone repository")) {
            errorMessage +=
              "\n\nThis could be because:\n" +
              "- The repository URL is incorrect\n" +
              "- The repository is private\n" +
              "- The repository is too large\n\n" +
              "Try using a smaller, public repository.";
          } else if (errorMessage.includes("list index out of range")) {
            errorMessage =
              "No processable text files found in the repository. " +
              "Please try a repository with Markdown, text, or code files.";
          }

          setInitError(errorMessage);
          setInitAttempts((prev) => prev + 1);
        } finally {
          setIsInitializing(false);
        }
      }
    };

    initializeGitHubProvider();
  }, [
    ragProvider,
    githubRepoUrl,
    githubRepoBranch,
    customConfig.apiBaseUrl,
    isInitialized,
    initAttempts,
  ]);

  // Function to retry initialization
  const handleRetry = () => {
    setInitAttempts(0);
    setIsInitialized(false);
  };

  return (
    <div className={className}>
      {initError && (
        <div className="alert alert-error mb-4">
          <div>
            <span className="font-bold">
              Error initializing GitHub provider:
            </span>
            <pre className="whitespace-pre-wrap text-sm mt-2">{initError}</pre>
            {initAttempts < 3 && (
              <button
                className="btn btn-sm btn-outline mt-2"
                onClick={handleRetry}
                disabled={isInitializing}
              >
                Retry
              </button>
            )}
          </div>
        </div>
      )}

      <TriggerButton
        onClick={() => setIsOpen(true)}
        className={buttonClassName}
        text={isInitializing ? "Initializing..." : buttonText}
        disabled={isInitializing}
      />
      <Modal
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        title={modalTitle}
      >
        <Chat config={customConfig} />
      </Modal>
    </div>
  );
};
