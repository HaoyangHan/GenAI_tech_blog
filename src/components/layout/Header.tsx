'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Upload, ChevronDown, User } from 'lucide-react';
import { BLOG_CATEGORIES } from '@/types';
import { categoryToSlug } from '@/lib/utils';

export default function Header() {
  const [showCategories, setShowCategories] = useState(false);

  const validCategories = BLOG_CATEGORIES.filter(cat => cat !== 'All' && cat !== 'Uncategorized');

  return (
    <header className="sticky top-0 bg-white/95 backdrop-blur-sm border-b border-gray-200 z-50">
      <div className="max-w-6xl mx-auto px-6 py-4">
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
            
            {/* Categories Dropdown */}
            <div className="relative">
              <button
                onClick={() => setShowCategories(!showCategories)}
                className="flex items-center space-x-1 text-gray-600 hover:text-gray-900 transition-colors font-medium"
              >
                <span>Categories</span>
                <ChevronDown size={16} className={`transform transition-transform ${showCategories ? 'rotate-180' : ''}`} />
              </button>
              
              {showCategories && (
                <div className="absolute top-full left-0 mt-2 w-64 bg-white rounded-lg shadow-lg border border-gray-200 py-2 z-50">
                  {validCategories.map((category) => (
                    <Link
                      key={category}
                      href={`/category/${categoryToSlug(category)}`}
                      className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 hover:text-gray-900"
                      onClick={() => setShowCategories(false)}
                    >
                      {category}
                    </Link>
                  ))}
                </div>
              )}
            </div>

            <Link 
              href="/about" 
              className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors font-medium"
            >
              <User size={16} />
              <span>About</span>
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
      
      {/* Overlay to close dropdown */}
      {showCategories && (
        <div 
          className="fixed inset-0 z-40" 
          onClick={() => setShowCategories(false)}
        />
      )}
    </header>
  );
} 