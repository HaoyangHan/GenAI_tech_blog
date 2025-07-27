'use client';

import { useState, useEffect } from 'react';
import Header from '@/components/layout/Header';
import CategoryFilter from '@/components/ui/CategoryFilter';
import BlogPostCard from '@/components/ui/BlogPostCard';
import { BlogPost, BlogCategory, FOUNDATION_CATEGORIES } from '@/types';
import Link from 'next/link';

export default function FoundationsKnowledgePage() {
  const [posts, setPosts] = useState<BlogPost[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<BlogCategory>('All');
  const [filteredPosts, setFilteredPosts] = useState<BlogPost[]>([]);

  useEffect(() => {
    // Load foundation posts from API
    const loadPosts = async () => {
      try {
        const response = await fetch('/api/posts?knowledgeBase=foundations');
        if (response.ok) {
          const allPosts = await response.json();
          const postsWithDates = allPosts.map((post: any) => ({
            ...post,
            date: new Date(post.date),
          }));
          setPosts(postsWithDates);
        }
      } catch (error) {
        console.error('Error loading foundation posts:', error);
      }
    };
    loadPosts();
  }, []);

  useEffect(() => {
    // Filter posts when category changes
    const loadFilteredPosts = async () => {
      try {
        const url = selectedCategory === 'All' 
          ? '/api/posts?knowledgeBase=foundations'
          : `/api/posts?knowledgeBase=foundations&category=${encodeURIComponent(selectedCategory)}`;
        
        const response = await fetch(url);
        if (response.ok) {
          const filtered = await response.json();
          const postsWithDates = filtered.map((post: any) => ({
            ...post,
            date: new Date(post.date),
          }));
          setFilteredPosts(postsWithDates);
        }
      } catch (error) {
        console.error('Error filtering foundation posts:', error);
      }
    };
    loadFilteredPosts();
  }, [selectedCategory]);

  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      <main className="max-w-4xl mx-auto px-6 py-12">
        {/* Breadcrumb */}
        <nav className="mb-6">
          <div className="flex items-center space-x-2 text-sm text-gray-500">
            <Link href="/" className="hover:text-purple-600">Home</Link>
            <span>/</span>
            <span className="text-gray-900">Data Science Foundations</span>
          </div>
        </nav>

        {/* Page Title */}
        <section className="mb-12">
          <h1 className="text-4xl font-bold text-purple-600 mb-4">
            Data Science Foundations
          </h1>
          <p className="text-lg text-gray-600 mb-8">
            Essential mathematical and statistical foundations for modern data science and machine learning. 
            Deep dive into the theoretical underpinnings of LLMs, traditional ML algorithms, probability theory, 
            and advanced optimization techniques with comprehensive proofs and implementations.
          </p>

          {/* Comprehensive Learning Path */}
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 p-6 rounded-lg border border-purple-200 mb-8">
            <h2 className="text-2xl font-semibold text-gray-900 mb-4 flex items-center">
              <svg className="w-6 h-6 text-purple-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
              Complete Foundations Curriculum
            </h2>
            <p className="text-gray-600 mb-6">Master data science and LLM foundations through our structured 41-article curriculum. Progress from mathematical fundamentals to cutting-edge AI concepts.</p>
            
            <div className="grid lg:grid-cols-2 gap-8">
              {/* LLM & AI Track */}
              <div className="bg-white p-6 rounded-lg shadow-sm border border-purple-100">
                <h3 className="text-xl font-semibold text-purple-900 mb-4 flex items-center">
                  <svg className="w-5 h-5 text-purple-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  LLM & AI Foundations (33 Articles)
                </h3>
                
                <div className="space-y-4">
                  {/* LLM Base Knowledge */}
                  <div className="border-l-4 border-purple-300 pl-4">
                    <h4 className="font-semibold text-purple-800 mb-2">1. LLM Base Knowledge (10 articles)</h4>
                    <p className="text-sm text-gray-600 mb-2">Foundations, tokenization, activations, language models, embeddings</p>
                    <div className="flex flex-wrap gap-1">
                      <span className="bg-purple-100 text-purple-700 px-2 py-1 rounded text-xs">Foundations</span>
                      <span className="bg-purple-100 text-purple-700 px-2 py-1 rounded text-xs">Tokenization</span>
                      <span className="bg-purple-100 text-purple-700 px-2 py-1 rounded text-xs">Activations</span>
                      <span className="bg-purple-100 text-purple-700 px-2 py-1 rounded text-xs">Embeddings</span>
                    </div>
                  </div>

                  {/* Model Architecture */}
                  <div className="border-l-4 border-blue-300 pl-4">
                    <h4 className="font-semibold text-blue-800 mb-2">2. LLM Model Architecture (9 articles)</h4>
                    <p className="text-sm text-gray-600 mb-2">Attention mechanisms, transformers, MOE, LLAMA models, decoding</p>
                    <div className="flex flex-wrap gap-1">
                      <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs">Attention</span>
                      <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs">Transformers</span>
                      <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs">MOE</span>
                      <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs">LLAMA</span>
                    </div>
                  </div>

                  {/* Training & Fine-tuning */}
                  <div className="border-l-4 border-green-300 pl-4">
                    <h4 className="font-semibold text-green-800 mb-2">3. Training & Fine-tuning (8 articles)</h4>
                    <p className="text-sm text-gray-600 mb-2">Training data, workflows, LoRA, adapter tuning, optimization</p>
                    <div className="flex flex-wrap gap-1">
                      <span className="bg-green-100 text-green-700 px-2 py-1 rounded text-xs">Training Data</span>
                      <span className="bg-green-100 text-green-700 px-2 py-1 rounded text-xs">LoRA</span>
                      <span className="bg-green-100 text-green-700 px-2 py-1 rounded text-xs">Adapters</span>
                    </div>
                  </div>

                  {/* RAG & Evaluation */}
                  <div className="border-l-4 border-orange-300 pl-4">
                    <h4 className="font-semibold text-orange-800 mb-2">4. RAG & Evaluation (6 articles)</h4>
                    <p className="text-sm text-gray-600 mb-2">RAG basics, agentic RAG, hallucination detection, evaluation methods</p>
                    <div className="flex flex-wrap gap-1">
                      <span className="bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs">RAG Theory</span>
                      <span className="bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs">Hallucination</span>
                      <span className="bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs">Evaluation</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Traditional ML Track */}
              <div className="bg-white p-6 rounded-lg shadow-sm border border-purple-100">
                <h3 className="text-xl font-semibold text-purple-900 mb-4 flex items-center">
                  <svg className="w-5 h-5 text-purple-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                  </svg>
                  Traditional ML & Statistics (8 Articles)
                </h3>
                
                <div className="space-y-4">
                  {/* Mathematical Foundations */}
                  <div className="border-l-4 border-indigo-300 pl-4">
                    <h4 className="font-semibold text-indigo-800 mb-2">5. Probability & Statistics (4 articles)</h4>
                    <p className="text-sm text-gray-600 mb-2">Mathematical expectations, distributions, imbalanced data, metrics</p>
                    <div className="flex flex-wrap gap-1">
                      <span className="bg-indigo-100 text-indigo-700 px-2 py-1 rounded text-xs">Probability</span>
                      <span className="bg-indigo-100 text-indigo-700 px-2 py-1 rounded text-xs">Distributions</span>
                      <span className="bg-indigo-100 text-indigo-700 px-2 py-1 rounded text-xs">Metrics</span>
                    </div>
                  </div>

                  {/* Practical Tools */}
                  <div className="border-l-4 border-pink-300 pl-4">
                    <h4 className="font-semibold text-pink-800 mb-2">6. ML Tools & Practice (4 articles)</h4>
                    <p className="text-sm text-gray-600 mb-2">Pandas operations, scikit-learn, classification metrics, ML concepts</p>
                    <div className="flex flex-wrap gap-1">
                      <span className="bg-pink-100 text-pink-700 px-2 py-1 rounded text-xs">Pandas</span>
                      <span className="bg-pink-100 text-pink-700 px-2 py-1 rounded text-xs">Scikit-learn</span>
                      <span className="bg-pink-100 text-pink-700 px-2 py-1 rounded text-xs">Classification</span>
                    </div>
                  </div>

                  {/* Learning Path Guide */}
                  <div className="bg-gradient-to-r from-purple-100 to-pink-100 p-4 rounded-lg mt-6">
                    <h4 className="font-semibold text-purple-900 mb-2">🎯 Recommended Learning Path</h4>
                    <ol className="text-sm text-purple-800 space-y-1">
                      <li><strong>1.</strong> Start with Traditional ML for solid foundations</li>
                      <li><strong>2.</strong> Move to LLM Base Knowledge</li>
                      <li><strong>3.</strong> Study Model Architecture in detail</li>
                      <li><strong>4.</strong> Explore Training & Fine-tuning techniques</li>
                      <li><strong>5.</strong> Master RAG & Evaluation methods</li>
                    </ol>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-6 p-4 bg-purple-100 rounded-lg">
              <p className="text-sm text-purple-800">
                <strong>💡 Study Tip:</strong> Each article includes mathematical proofs, code implementations, and interview questions. 
                We recommend studying 2-3 articles per week for optimal retention and understanding.
              </p>
            </div>
          </div>
          
          {/* Enhanced Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <div className="bg-purple-50 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-purple-600">41</div>
              <div className="text-sm text-purple-700">Expert Articles</div>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-purple-600">6</div>
              <div className="text-sm text-purple-700">Study Tracks</div>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-purple-600">∞</div>
              <div className="text-sm text-purple-700">Math Proofs</div>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-purple-600">100%</div>
              <div className="text-sm text-purple-700">Interview Ready</div>
            </div>
          </div>

          {/* Quick Access Links */}
          <div className="bg-purple-50 rounded-lg p-6 mb-8">
            <h3 className="text-lg font-semibold text-purple-900 mb-4">🚀 Quick Start Guides</h3>
            <div className="grid md:grid-cols-3 gap-4">
              <Link href="/knowledge/foundations?category=Traditional ML" className="bg-white p-4 rounded-lg border hover:border-purple-300 transition-colors">
                <div className="font-medium text-purple-800 mb-2">👩‍🔬 Start with Traditional ML</div>
                <div className="text-sm text-purple-600">Begin with probability, statistics, and classical machine learning foundations</div>
              </Link>
              <Link href="/knowledge/foundations?category=LLM Base Knowledge" className="bg-white p-4 rounded-lg border hover:border-purple-300 transition-colors">
                <div className="font-medium text-purple-800 mb-2">🧠 Dive into LLM Basics</div>
                <div className="text-sm text-purple-600">Explore tokenization, embeddings, and language model fundamentals</div>
              </Link>
              <Link href="/knowledge/foundations?category=LLM Model Architecture" className="bg-white p-4 rounded-lg border hover:border-purple-300 transition-colors">
                <div className="font-medium text-purple-800 mb-2">🏗️ Master Architectures</div>
                <div className="text-sm text-purple-600">Study attention mechanisms, transformers, and modern model designs</div>
              </Link>
            </div>
          </div>
        </section>

        {/* Category Filter */}
        <CategoryFilter 
          selectedCategory={selectedCategory}
          onCategoryChange={setSelectedCategory}
          availableCategories={FOUNDATION_CATEGORIES}
        />

        {/* Blog Posts */}
        <section className="space-y-0">
          {filteredPosts.length === 0 ? (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">No Posts Found</h3>
              <p className="text-gray-600">
                {selectedCategory === 'All' 
                  ? 'No foundation posts found. Check back soon for new content!' 
                  : `No posts found in "${selectedCategory}" category.`
                }
              </p>
            </div>
          ) : (
            filteredPosts.map((post) => (
              <BlogPostCard key={post.id} post={post} />
            ))
          )}
        </section>
      </main>
    </div>
  );
} 