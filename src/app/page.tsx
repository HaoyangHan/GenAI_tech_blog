'use client';

import Header from '@/components/layout/Header';
import Link from 'next/link';
import { KnowledgeBaseInfo } from '@/types';

const knowledgeBases: KnowledgeBaseInfo[] = [
  {
    id: 'rag',
    title: 'RAG Implementation Hub',
    description: 'Complete journey from business objectives to production deployment of Retrieval-Augmented Generation systems. Explore real-world implementation details, engineering decisions, performance optimizations, and best practices for building scalable RAG applications.',
    categories: ['Business Objective', 'Engineering Architecture', 'Ingestion', 'Retrieval', 'Generation', 'Evaluation', 'Prompt Tuning', 'Agentic Workflow'],
    path: '/knowledge/rag'
  },
  {
    id: 'foundations',
    title: 'Data Science Foundations',
    description: 'Essential mathematical and statistical foundations for modern data science and machine learning. Deep dive into the theoretical underpinnings of LLMs, traditional ML algorithms, probability theory, and advanced optimization techniques with comprehensive proofs and implementations.',
    categories: ['Statistical Deep Dive'],
    path: '/knowledge/foundations'
  }
];

export default function HomePage() {
  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      <main className="max-w-6xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <section className="text-center mb-16">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            GenAI Knowledge Hub
          </h1>
          <p className="text-xl text-gray-600 max-w-4xl mx-auto leading-relaxed">
            A comprehensive resource for modern AI practitioners, covering both the 
            <span className="font-semibold text-blue-600"> practical implementation of RAG systems</span> and the 
            <span className="font-semibold text-purple-600"> theoretical foundations of data science</span>.
            Whether you&apos;re building production AI systems or deepening your mathematical understanding, 
            find the knowledge you need to excel.
          </p>
        </section>

        {/* Knowledge Bases Grid */}
        <section className="grid md:grid-cols-2 gap-8 mb-16">
          {knowledgeBases.map((kb) => (
            <div key={kb.id} className="bg-white border border-gray-200 rounded-xl p-8 shadow-sm hover:shadow-lg transition-shadow duration-300">
              <div className="mb-6">
                <h2 className={`text-2xl font-bold mb-4 ${
                  kb.id === 'rag' ? 'text-blue-600' : 'text-purple-600'
                }`}>
                  {kb.title}
                </h2>
                <p className="text-gray-700 leading-relaxed mb-6">
                  {kb.description}
                </p>
              </div>

              {/* Categories Preview */}
              <div className="mb-6">
                <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
                  Topics Covered
                </h3>
                <div className="flex flex-wrap gap-2">
                  {kb.categories.slice(0, 4).map((category) => (
                    <span 
                      key={category} 
                      className={`px-3 py-1 text-xs font-medium rounded-full ${
                        kb.id === 'rag' 
                          ? 'bg-blue-50 text-blue-700 border border-blue-200' 
                          : 'bg-purple-50 text-purple-700 border border-purple-200'
                      }`}
                    >
                      {category}
                    </span>
                  ))}
                  {kb.categories.length > 4 && (
                    <span className="px-3 py-1 text-xs font-medium rounded-full bg-gray-50 text-gray-600 border border-gray-200">
                      +{kb.categories.length - 4} more
                    </span>
                  )}
                </div>
              </div>

              {/* CTA Button */}
              <Link 
                href={kb.path}
                className={`inline-flex items-center justify-center w-full px-6 py-3 text-sm font-medium text-white rounded-lg transition-colors duration-200 ${
                  kb.id === 'rag'
                    ? 'bg-blue-600 hover:bg-blue-700'
                    : 'bg-purple-600 hover:bg-purple-700'
                }`}
              >
                Explore {kb.title}
                <svg className="ml-2 w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </Link>
            </div>
          ))}
        </section>

        {/* Features Section */}
        <section className="bg-gray-50 rounded-2xl p-8 mb-16">
          <h2 className="text-3xl font-bold text-gray-900 text-center mb-8">
            Why This Knowledge Hub?
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Practical Implementation</h3>
              <p className="text-gray-600">Real-world code examples, architecture decisions, and production-ready solutions.</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Mathematical Rigor</h3>
              <p className="text-gray-600">Deep theoretical foundations with proofs, derivations, and mathematical insights.</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Cutting-Edge Knowledge</h3>
              <p className="text-gray-600">Latest developments in AI/ML with insights from hands-on experience.</p>
            </div>
          </div>
        </section>

        {/* Quick Stats */}
        <section className="text-center">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <div>
              <div className="text-3xl font-bold text-blue-600 mb-2">8+</div>
              <div className="text-sm text-gray-600">RAG Topics</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-purple-600 mb-2">40+</div>
              <div className="text-sm text-gray-600">Foundation Articles</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-green-600 mb-2">100%</div>
              <div className="text-sm text-gray-600">Practical Focus</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-orange-600 mb-2">∞</div>
              <div className="text-sm text-gray-600">Learning Potential</div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
} 