import Link from 'next/link';
import { Upload } from 'lucide-react';

export default function Header() {
  return (
    <header className="sticky top-0 bg-white/95 backdrop-blur-sm border-b border-gray-200 z-50">
      <div className="max-w-4xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo / Brand */}
          <Link 
            href="/" 
            className="text-xl font-bold text-gray-900 hover:text-gray-700 transition-colors"
          >
            GenAI Tech Blog
          </Link>

          {/* Navigation */}
          <nav className="flex items-center space-x-6">
            <Link 
              href="/" 
              className="text-gray-600 hover:text-gray-900 transition-colors font-medium"
            >
              Articles
            </Link>
            <Link 
              href="/upload" 
              className="flex items-center space-x-2 bg-gray-900 text-white px-4 py-2 rounded-lg hover:bg-gray-800 transition-colors font-medium"
            >
              <Upload size={16} />
              <span>Upload</span>
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
} 