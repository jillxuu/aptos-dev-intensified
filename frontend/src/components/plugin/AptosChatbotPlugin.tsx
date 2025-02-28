import { useState } from "react";
import { Modal } from "./Modal";
import { TriggerButton } from "./TriggerButton";
import Chat from "../chat/Chat";
import { config as defaultConfig } from "../../config";

interface AptosChatbotPluginProps {
  className?: string;
  buttonText?: string;
  buttonClassName?: string;
  modalTitle?: string;
  apiUrl?: string;
}

/**
 * AptosChatbotPlugin - A standalone plugin component that can be embedded anywhere
 *
 * @param className - Additional class names for the container
 * @param buttonText - Custom text for the trigger button
 * @param buttonClassName - Additional class names for the button
 * @param modalTitle - Custom title for the modal header
 * @param apiUrl - Optional custom API URL to override the default backend URL
 */
export const AptosChatbotPlugin: React.FC<AptosChatbotPluginProps> = ({
  className = "",
  buttonText = "Ask Aptos AI",
  buttonClassName = "",
  modalTitle = "Ask Aptos AI",
  apiUrl,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  // Create a custom config if apiUrl is provided
  const config = apiUrl
    ? { ...defaultConfig, apiBaseUrl: apiUrl }
    : defaultConfig;

  return (
    <div className={className}>
      <TriggerButton
        onClick={() => setIsOpen(true)}
        className={buttonClassName}
        text={buttonText}
      />
      <Modal
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        title={modalTitle}
      >
        <Chat config={config} />
      </Modal>
    </div>
  );
};
