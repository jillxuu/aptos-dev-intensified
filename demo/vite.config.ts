import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@aptos/chatbot-core': resolve(__dirname, '../../packages/chatbot-core/src'),
      '@aptos/chatbot-react': resolve(__dirname, '../../packages/chatbot-react/src'),
      '@aptos/chatbot-ui-tailwind': resolve(__dirname, '../../packages/chatbot-ui-tailwind/src'),
    },
  },
});
