import { type ReactElement } from 'react';
import { ChatSidebar as BaseChatSidebar } from '@aptos/chatbot-ui-base';
import { ChatSidebarProps } from '@aptos/chatbot-react';

export function ChatSidebar(props: ChatSidebarProps): ReactElement {
  return (
    <BaseChatSidebar
      {...props}
      className="w-80 border-r border-border-primary bg-background-secondary flex flex-col"
    />
  );
}
