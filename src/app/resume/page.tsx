'use client';

import Header from '@/components/layout/Header';
import Link from 'next/link';
import { ArrowLeft, Download, ExternalLink } from 'lucide-react';

export default function ResumePage() {
  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      <main className="max-w-6xl mx-auto px-6 py-12">
        {/* Breadcrumb and Actions */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8">
          <nav className="mb-4 sm:mb-0">
            <div className="flex items-center space-x-2 text-sm text-gray-500">
              <Link href="/" className="hover:text-blue-600">Home</Link>
              <span>/</span>
              <Link href="/about" className="hover:text-blue-600">About</Link>
              <span>/</span>
              <span className="text-gray-900">Resume</span>
            </div>
          </nav>
          
          <div className="flex space-x-3">
            <Link
              href="/about"
              className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to About
            </Link>
            <a
              href="/assets/Haoyang Han DS resume.pdf"
              download="Haoyang_Han_Resume.pdf"
              className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Download className="w-4 h-4 mr-2" />
              Download PDF
            </a>
          </div>
        </div>

        {/* Page Title */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Resume
          </h1>
          <p className="text-lg text-gray-600">
            Senior Data Scientist | GenAI & Machine Learning Expert
          </p>
        </div>

        {/* Resume Container */}
        <div className="bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden">
          {/* PDF Viewer */}
          <div className="relative w-full" style={{ height: '800px' }}>
            <iframe
              src="/assets/Haoyang Han DS resume.pdf"
              className="w-full h-full border-0"
              title="Haoyang Han Resume"
            />
          </div>
        </div>

        {/* Fallback Download Section */}
        <div className="mt-8 text-center">
          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Having trouble viewing the resume?
            </h3>
            <p className="text-gray-600 mb-4">
              If the PDF doesn&apos;t display properly in your browser, you can download it directly.
            </p>
            <a
              href="/assets/Haoyang Han DS resume.pdf"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center px-6 py-3 text-sm font-medium text-blue-600 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors"
            >
              <ExternalLink className="w-4 h-4 mr-2" />
              Open in New Tab
            </a>
          </div>
        </div>

        {/* Contact Information */}
        <div className="mt-12 text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Get In Touch</h2>
          <p className="text-gray-600 mb-4">
            Interested in discussing opportunities or have questions about my experience?
          </p>
          <a
            href="mailto:haoyanghan1996@gmail.com?subject=from%20genaiknowledge.info%20-%20Resume%20Inquiry"
            className="inline-flex items-center px-6 py-3 text-white bg-green-600 rounded-lg hover:bg-green-700 transition-colors"
          >
            Contact Me
          </a>
        </div>
      </main>
    </div>
  );
} 