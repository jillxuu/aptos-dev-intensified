import { type ReactElement } from 'react';
import { ChatWidget as BaseChatWidget } from '@aptos/chatbot-ui-base';
import { ChatWidgetProps } from '@aptos/chatbot-react';

export function ChatWidget(props: ChatWidgetProps): ReactElement {
  return (
    <div className="flex h-full">
      <BaseChatWidget
        {...props}
        className="flex-1 flex flex-col rounded-lg overflow-hidden shadow-md"
        messageClassName="prose max-w-none p-4 rounded-lg bg-background-secondary last:bg-background-elevated text-text-primary"
        inputClassName="p-4 border-t border-border-primary"
      />
    </div>
  );
}
