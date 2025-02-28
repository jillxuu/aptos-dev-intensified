import { AptosChatbotPlugin } from "./components/plugin";

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
      </div>
    </div>
  );
}

export default App;
