import { AptosChatbotPlugin } from "../components/plugin";

const PluginTest = () => {
  return (
    <div className="min-h-screen p-8 bg-base-200">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Plugin Test Page</h1>
        <div className="space-y-8">
          <div>
            <h2 className="text-xl font-semibold mb-4">
              Try the Aptos AI Plugin
            </h2>
            <p className="mb-4 text-base-content/70">
              Click the button below to open the Aptos AI chat interface in a
              modal.
            </p>
            <AptosChatbotPlugin />
          </div>
        </div>
      </div>
    </div>
  );
};

export default PluginTest;
