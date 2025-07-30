import type { Metadata } from 'next';
import Script from 'next/script';
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
      <head>
        {/* Google Analytics */}
        <Script
          async
          src="https://www.googletagmanager.com/gtag/js?id=G-C2XHG5LGKX"
          strategy="afterInteractive"
        />
        <Script id="google-analytics" strategy="afterInteractive">
          {`
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', 'G-C2XHG5LGKX');
          `}
        </Script>
      </head>
      <body className="font-sans">
        {children}
      </body>
    </html>
  );
} 