interface TriggerButtonProps {
  onClick: () => void;
  className?: string;
  text?: string;
}

export const TriggerButton: React.FC<TriggerButtonProps> = ({
  onClick,
  className = "",
  text = "Ask Aptos AI",
}) => {
  return (
    <button
      onClick={onClick}
      className={`flex items-center space-x-2 px-4 py-2 bg-[#1C1C1E] text-white rounded-lg hover:bg-[#2C2C2E] transition-colors ${className}`}
    >
      <img src="/aptos-logo.svg" alt="Aptos Logo" className="w-5 h-5" />
      <span>{text}</span>
    </button>
  );
};
