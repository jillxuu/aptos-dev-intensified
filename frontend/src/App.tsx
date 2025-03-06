import { AptosChatbotPlugin } from "./components/plugin";
import { Link } from "react-router-dom";

function App() {
  return (
    <div className="min-h-screen p-8 bg-base-200">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">
          Aptos AI Chatbot Plugin Demo
        </h1>

        <div className="bg-base-100 p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4">Default Plugin</h2>
          <p className="mb-4 text-base-content/70">
            Click the button below to open the Aptos AI chat interface.
          </p>
          <AptosChatbotPlugin />
        </div>

        <div className="bg-base-100 p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4">Examples</h2>
          <p className="mb-4 text-base-content/70">
            Check out these examples to see how you can customize the plugin:
          </p>
          <div className="flex flex-col gap-2">
            <Link to="/github-rag" className="btn btn-outline btn-primary">
              GitHub Repository as Knowledge Base
            </Link>
          </div>
        </div>

        <div className="bg-base-100 p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">Integration</h2>
          <p className="mb-4 text-base-content/70">
            To integrate this plugin into your own application, install the
            package and import the component:
          </p>
          <pre className="bg-base-300 p-4 rounded-md overflow-x-auto mb-4">
            <code>npm install aptos-chatbot-plugin</code>
          </pre>
          <pre className="bg-base-300 p-4 rounded-md overflow-x-auto">
            <code>{`import { AptosChatbotPlugin } from "aptos-chatbot-plugin";

function YourComponent() {
  return (
    <div>
      <AptosChatbotPlugin />
    </div>
  );
}`}</code>
          </pre>
        </div>
      </div>
    </div>
  );
}

export default App;
