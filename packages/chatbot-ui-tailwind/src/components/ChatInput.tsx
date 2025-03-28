import { type ReactElement } from 'react';
import { ChatInput as BaseChatInput } from '@aptos/chatbot-ui-base';
import { ChatInputProps } from '@aptos/chatbot-react';
export function ChatInput(props: ChatInputProps): ReactElement {
  return (
    <BaseChatInput
      {...props}
      className="p-4 border-t border-gray-200 flex gap-2"
      placeholder="Type a message..."
    />
  );
}
