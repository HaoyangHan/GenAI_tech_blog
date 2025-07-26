import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'GenAI Tech Blog - RAG Implementation Details',
  description: 'A technical blog exploring RAG implementation, from business objectives to evaluation metrics.',
  keywords: ['RAG', 'LLM', 'AI', 'Machine Learning', 'Vector Database', 'Retrieval'],
  authors: [{ name: 'GenAI Tech Blog' }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-sans">
        {children}
      </body>
    </html>
  );
} 