import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';

interface CopyButtonProps {
  text: string;
  className?: string;
  size?: number;
}

export default function CopyButton({ text, className = '', size = 16 }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  return (
    <button
      onClick={handleCopy}
      className={`
        inline-flex items-center justify-center
        p-1.5 rounded-md
        bg-gray-700 hover:bg-gray-600
        text-gray-300 hover:text-white
        transition-all duration-200
        opacity-0 group-hover:opacity-100
        ${className}
      `}
      title={copied ? 'Copied!' : 'Copy to clipboard'}
      aria-label={copied ? 'Copied!' : 'Copy to clipboard'}
    >
      {copied ? (
        <Check size={size} className="text-green-400" />
      ) : (
        <Copy size={size} />
      )}
    </button>
  );
} 