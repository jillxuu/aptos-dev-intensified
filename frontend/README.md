# Aptos Chatbot Plugin

A React component that provides an embeddable Aptos AI chatbot for any React application.

## Installation

```bash
npm install aptos-chatbot-plugin
# or
yarn add aptos-chatbot-plugin
# or
pnpm add aptos-chatbot-plugin
```

## Usage

```jsx
import { AptosChatbotPlugin } from "aptos-chatbot-plugin";

function App() {
  return (
    <div className="my-app">
      <h1>My Application</h1>

      {/* Basic usage with default hosted backend */}
      <AptosChatbotPlugin />

      {/* With customization */}
      <AptosChatbotPlugin
        buttonText="Ask Aptos AI"
        modalTitle="Aptos AI Assistant"
        buttonClassName="my-custom-button-class"
        className="my-custom-container-class"
      />

      {/* With custom backend URL (advanced usage) */}
      <AptosChatbotPlugin apiUrl="https://your-custom-backend-url.com/api" />
    </div>
  );
}
```

## Props

| Prop              | Type   | Default        | Description                                                 |
| ----------------- | ------ | -------------- | ----------------------------------------------------------- |
| `buttonText`      | string | "Ask Aptos AI" | Text displayed on the trigger button                        |
| `modalTitle`      | string | "Ask Aptos AI" | Title displayed in the modal header                         |
| `buttonClassName` | string | ""             | Additional CSS classes for the button                       |
| `className`       | string | ""             | Additional CSS classes for the container                    |
| `apiUrl`          | string | undefined      | Optional custom API URL to override the default backend URL |

## Backend Configuration

The Aptos Chatbot Plugin comes with a pre-configured backend URL that connects to our hosted service. This means you don't need to set up or host your own backend to use the plugin.

### Default Hosted Backend

By default, the plugin connects to our hosted backend service, which provides:

- RAG implementation using LangChain
- Support for both Claude and ChatGPT models
- Up-to-date Aptos documentation and resources
- Reliable and scalable infrastructure

### Custom Backend (Advanced)

If you need to use your own backend:

1. Set up your own backend server using the code from our [GitHub repository](https://github.com/yourusername/aptos-dev-intensified)
2. Pass your custom backend URL using the `apiUrl` prop:

```jsx
<AptosChatbotPlugin apiUrl="https://your-custom-backend-url.com/api" />
```

## Environment Variables (Optional)

You can override the default backend URL by setting an environment variable in your application:

```
VITE_API_URL=https://your-backend-api.com
```

This is useful for development or if you want to configure the URL at build time.

## Dependencies

This package has the following peer dependencies:

- React 18+
- React DOM 18+
- Tailwind CSS (for styling)

## License

MIT
