import { useState, useEffect } from "react";
import { AptosChatbotPlugin } from "../components/plugin";
import { apiService } from "../services";

// Default repository for the example - using a smaller repo
const DEFAULT_REPO = "https://github.com/aptos-labs/aptos-ts-sdk";

const GitHubRagExample = () => {
  const [repoUrl, setRepoUrl] = useState<string>("");
  const [branch, setBranch] = useState<string>("main");
  const [isInitializing, setIsInitializing] = useState<boolean>(false);
  const [isInitialized, setIsInitialized] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // For the direct integration example
  const [, setDirectExampleInitialized] = useState<boolean>(false);
  const [directExampleError, setDirectExampleError] = useState<string | null>(
    null,
  );
  const [directExampleRepo] = useState<string>(
    "https://github.com/aptos-labs/aptos-ts-sdk",
  );
  const [directExampleBranch] = useState<string>("main");

  // Initialize the direct example repository on component mount
  useEffect(() => {
    const initializeDirectExample = async () => {
      try {
        await apiService.initializeGitHubRepo({
          repo_url: directExampleRepo,
          branch: directExampleBranch,
        });
        setDirectExampleInitialized(true);
      } catch (err: any) {
        console.error("Error initializing direct example repository:", err);
        setDirectExampleError(
          err.response?.data?.detail ||
            "Failed to initialize the example repository. You may still try with your own repository.",
        );
      }
    };

    initializeDirectExample();
  }, [directExampleRepo, directExampleBranch]);

  const handleInitialize = async () => {
    if (!repoUrl) {
      setError("Please enter a GitHub repository URL");
      return;
    }

    setIsInitializing(true);
    setError(null);
    setSuccess(null);

    try {
      await apiService.initializeGitHubRepo({
        repo_url: repoUrl,
        branch: branch,
      });

      setIsInitialized(true);
      setSuccess(
        "GitHub repository initialized successfully! You can now use the chatbot.",
      );
    } catch (err: any) {
      console.error("Error initializing GitHub Knowledge provider:", err);
      setError(
        err.response?.data?.detail ||
          "Failed to initialize GitHub Knowledge provider",
      );
    } finally {
      setIsInitializing(false);
    }
  };

  return (
    <div className="min-h-screen p-8 bg-base-200">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">
          GitHub Knowledge Base Example
        </h1>

        <div className="bg-base-100 p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4">
            Use a GitHub Repository as Knowledge Base
          </h2>
          <p className="mb-6 text-base-content/70">
            Enter a GitHub repository URL to use as the knowledge base for the
            chatbot. The chatbot will clone the repository, process its
            contents, and use it to answer questions.
          </p>

          <div className="form-control mb-4">
            <label className="label">
              <span className="label-text">GitHub Repository URL</span>
            </label>
            <input
              type="text"
              placeholder="https://github.com/username/repository"
              className="input input-bordered w-full"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
            />
            <label className="label">
              <span className="label-text-alt">Example: {DEFAULT_REPO}</span>
            </label>
          </div>

          <div className="form-control mb-6">
            <label className="label">
              <span className="label-text">Branch</span>
            </label>
            <input
              type="text"
              placeholder="main"
              className="input input-bordered w-full"
              value={branch}
              onChange={(e) => setBranch(e.target.value)}
            />
            <label className="label">
              <span className="label-text-alt">Default: main</span>
            </label>
          </div>

          <button
            className="btn btn-primary"
            onClick={handleInitialize}
            disabled={isInitializing || !repoUrl}
          >
            {isInitializing ? "Initializing..." : "Initialize Repository"}
          </button>

          {error && (
            <div className="alert alert-error mt-4">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="stroke-current shrink-0 h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span>{error}</span>
            </div>
          )}

          {success && (
            <div className="alert alert-success mt-4">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="stroke-current shrink-0 h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span>{success}</span>
            </div>
          )}
        </div>

        {isInitialized && (
          <div className="bg-base-100 p-6 rounded-lg shadow-md mb-8">
            <h2 className="text-xl font-semibold mb-4">
              Ask Questions About the Repository
            </h2>
            <p className="mb-6 text-base-content/70">
              Now you can ask questions about the content of the GitHub
              repository. The chatbot will use the repository as its knowledge
              base to answer your questions.
            </p>

            <div className="bg-base-300 p-4 rounded-md mb-6">
              <pre className="text-sm overflow-x-auto">
                {`<AptosChatbotPlugin 
  ragProvider="github" 
  githubRepoUrl="${repoUrl}"
  githubRepoBranch="${branch}"
  buttonText="Ask about ${repoUrl.split("/").pop() || "the repository"}" 
/>`}
              </pre>
            </div>

            <AptosChatbotPlugin
              ragProvider="github"
              githubRepoUrl={repoUrl}
              githubRepoBranch={branch}
              buttonText={`Ask about ${repoUrl.split("/").pop() || "the repository"}`}
            />
          </div>
        )}

        <div className="bg-base-100 p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4">
            Direct Integration Example
          </h2>
          <p className="mb-6 text-base-content/70">
            You can also directly integrate the chatbot with a GitHub repository
            without requiring the user to initialize it manually. Just provide
            the repository URL and branch as props:
          </p>

          <div className="bg-base-300 p-4 rounded-md mb-6">
            <pre className="text-sm overflow-x-auto">
              {`<AptosChatbotPlugin 
  ragProvider="github" 
  githubRepoUrl="${directExampleRepo}"
  githubRepoBranch="${directExampleBranch}"
  buttonText="Ask about Aptos TypeScript SDK" 
/>`}
            </pre>
          </div>

          {directExampleError && (
            <div className="alert alert-warning mb-4">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="stroke-current shrink-0 h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
              <span>{directExampleError}</span>
            </div>
          )}

          <AptosChatbotPlugin
            ragProvider="github"
            githubRepoUrl={directExampleRepo}
            githubRepoBranch={directExampleBranch}
            buttonText="Ask about Aptos TypeScript SDK"
          />
        </div>

        <div className="bg-base-100 p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">How It Works</h2>
          <ol className="list-decimal list-inside space-y-2 mb-4">
            <li>The system clones the GitHub repository you specified</li>
            <li>
              It processes the repository files and creates a vector store
            </li>
            <li>
              When you ask a question, it retrieves relevant content from the
              repository
            </li>
            <li>The AI generates a response based on the retrieved content</li>
          </ol>

          <p className="text-sm text-base-content/70 mt-6">
            Note: This is a demonstration of using a GitHub repository as a
            knowledge base. In a production environment, you might want to
            implement caching and periodic updates to ensure the knowledge base
            stays up-to-date with the repository.
          </p>
        </div>
      </div>
    </div>
  );
};

export default GitHubRagExample;
