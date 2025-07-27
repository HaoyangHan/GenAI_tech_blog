'use client';

import Header from '@/components/layout/Header';
import Link from 'next/link';
import { Brain, Code, Lightbulb, BookOpen, FileText, Github, Mail } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      <main className="max-w-4xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            About Me
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Senior Data Scientist specializing in GenAI & Traditional ML at Citi
          </p>
          <div className="mt-6 flex justify-center space-x-4">
            <a
              href="mailto:haoyanghan1996@gmail.com?subject=from%20genaiknowledge.info"
              className="inline-flex items-center px-4 py-2 text-sm font-medium text-blue-600 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors"
            >
              <Mail className="w-4 h-4 mr-2" />
              Contact Me
            </a>
            <Link
              href="/resume"
              className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-green-600 rounded-lg hover:bg-green-700 transition-colors"
            >
              <FileText className="w-4 h-4 mr-2" />
              View Resume
            </Link>
          </div>
        </div>

        {/* Personal Introduction */}
        <div className="mb-16">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-8 mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-6">My Journey</h2>
            <div className="prose prose-lg max-w-none text-gray-700 space-y-4">
              <p>
                Hi, I&apos;m <strong>Haoyang Han</strong>, and this website represents a culmination of my 5-year journey 
                as a Senior Data Scientist at Citi. My expertise spans from traditional NLP—where I fine-tuned 
                BERT-like models for specific tasks—to the cutting-edge GenAI domain, where I now focus on 
                prompt engineering, model selection, and evaluation.
              </p>
              <p>
                This knowledge hub contains essential insights from my RAG implementation experience and key 
                mathematical foundations that I believe every data scientist should master in the GenAI era.
              </p>
              <p>
                My methodology involves building comprehensive content skeletons and using sophisticated prompts 
                to generate detailed technical documentation. You can explore my approach and other projects on{' '}
                <a 
                  href="https://github.com/HaoyangHan" 
                  className="text-blue-600 underline hover:text-blue-800 inline-flex items-center" 
                  target="_blank" 
                  rel="noopener noreferrer"
                >
                  GitHub <Github className="w-4 h-4 ml-1" />
                </a>.
              </p>
              <div className="mt-6 p-4 bg-white rounded-lg border border-blue-200">
                <p className="text-sm text-blue-800 mb-2">
                  <strong>📄 Professional Background:</strong>
                </p>
                <p className="text-sm text-gray-700">
                  For a detailed overview of my experience, skills, and achievements, check out my{' '}
                  <Link href="/resume" className="text-blue-600 underline hover:text-blue-800 font-medium">
                    complete resume
                  </Link>.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* What You'll Find Here */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-12 text-center">What You&apos;ll Discover</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
              <div className="flex items-center mb-4">
                <div className="bg-blue-100 w-12 h-12 rounded-lg flex items-center justify-center mr-4">
                  <Brain className="w-6 h-6 text-blue-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900">Deep Technical Insights</h3>
              </div>
              <p className="text-gray-600">
                Mathematical foundations, architecture deep-dives, and implementation details 
                that bridge theory with practice in modern AI systems.
              </p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
              <div className="flex items-center mb-4">
                <div className="bg-green-100 w-12 h-12 rounded-lg flex items-center justify-center mr-4">
                  <Code className="w-6 h-6 text-green-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900">Production-Ready Code</h3>
              </div>
              <p className="text-gray-600">
                Real-world implementations, best practices, and hands-on guides for building 
                scalable RAG systems and AI applications.
              </p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
              <div className="flex items-center mb-4">
                <div className="bg-purple-100 w-12 h-12 rounded-lg flex items-center justify-center mr-4">
                  <Lightbulb className="w-6 h-6 text-purple-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900">Strategic Perspectives</h3>
              </div>
              <p className="text-gray-600">
                Business applications, evaluation frameworks, and insights from deploying 
                AI systems in enterprise environments.
              </p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
              <div className="flex items-center mb-4">
                <div className="bg-orange-100 w-12 h-12 rounded-lg flex items-center justify-center mr-4">
                  <BookOpen className="w-6 h-6 text-orange-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900">Curated Resources</h3>
              </div>
              <p className="text-gray-600">
                Essential papers, tools, and methodologies for continuous learning 
                in the rapidly evolving AI landscape.
              </p>
            </div>
          </div>
        </div>

        {/* Two Knowledge Areas */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">Knowledge Areas</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-8 rounded-2xl">
              <h3 className="text-2xl font-bold text-blue-900 mb-4">RAG Implementation Hub</h3>
              <p className="text-blue-800 mb-6">
                End-to-end journey from business objectives to production deployment. 
                Real engineering decisions, performance optimizations, and lessons learned.
              </p>
              <Link 
                href="/knowledge/rag"
                className="inline-flex items-center px-4 py-2 text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors"
              >
                Explore RAG Knowledge →
              </Link>
            </div>
            
            <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-8 rounded-2xl">
              <h3 className="text-2xl font-bold text-purple-900 mb-4">Data Science Foundations</h3>
              <p className="text-purple-800 mb-6">
                Mathematical foundations, statistical theory, and core ML concepts. 
                From attention mechanisms to traditional ML algorithms.
              </p>
              <Link 
                href="/knowledge/foundations"
                className="inline-flex items-center px-4 py-2 text-white bg-purple-600 rounded-lg hover:bg-purple-700 transition-colors"
              >
                Explore Foundations →
              </Link>
            </div>
          </div>
        </div>

        {/* Technical Note */}
        <div className="bg-gray-50 rounded-2xl p-8 text-center">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Technical Implementation</h3>
          <p className="text-gray-600 max-w-2xl mx-auto">
            This knowledge hub is built with Next.js 14, featuring advanced markdown processing 
            with LaTeX equations, syntax highlighting, and interactive elements. All content 
            is version-controlled and continuously updated with the latest insights.
          </p>
        </div>
      </main>
    </div>
  );
} 