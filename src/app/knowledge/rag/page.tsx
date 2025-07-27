'use client';

import { useState, useEffect } from 'react';
import Header from '@/components/layout/Header';
import CategoryFilter from '@/components/ui/CategoryFilter';
import BlogPostCard from '@/components/ui/BlogPostCard';
import { BlogPost, BlogCategory, RAG_CATEGORIES } from '@/types';
import Link from 'next/link';

export default function RAGKnowledgePage() {
  const [posts, setPosts] = useState<BlogPost[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<BlogCategory>('All');
  const [filteredPosts, setFilteredPosts] = useState<BlogPost[]>([]);

  useEffect(() => {
    // Load RAG posts from API
    const loadPosts = async () => {
      try {
        const response = await fetch('/api/posts?knowledgeBase=rag');
        if (response.ok) {
          const allPosts = await response.json();
          const postsWithDates = allPosts.map((post: any) => ({
            ...post,
            date: new Date(post.date),
          }));
          setPosts(postsWithDates);
        }
      } catch (error) {
        console.error('Error loading RAG posts:', error);
      }
    };
    loadPosts();
  }, []);

  useEffect(() => {
    // Filter posts when category changes
    const loadFilteredPosts = async () => {
      try {
        const url = selectedCategory === 'All' 
          ? '/api/posts?knowledgeBase=rag'
          : `/api/posts?knowledgeBase=rag&category=${encodeURIComponent(selectedCategory)}`;
        
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
        console.error('Error filtering RAG posts:', error);
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
            <Link href="/" className="hover:text-blue-600">Home</Link>
            <span>/</span>
            <span className="text-gray-900">RAG Implementation Hub</span>
          </div>
        </nav>

        {/* Page Title */}
        <section className="mb-12">
          <h1 className="text-4xl font-bold text-blue-600 mb-4">
            RAG Implementation Hub
          </h1>
          <p className="text-lg text-gray-600 mb-8">
            Complete journey from business objectives to production deployment of Retrieval-Augmented Generation systems. 
            Explore real-world implementation details, engineering decisions, and performance optimizations.
          </p>

          {/* Learning Path Table of Contents */}
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg border border-blue-200 mb-8">
            <h2 className="text-2xl font-semibold text-gray-900 mb-4 flex items-center">
              <svg className="w-6 h-6 text-blue-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              Complete Learning Path
            </h2>
            <p className="text-gray-600 mb-6">Follow this structured path to master production-ready RAG systems from concept to deployment.</p>
            
            <div className="overflow-x-auto">
              <table className="w-full border-collapse bg-white rounded-lg shadow-sm">
                <thead>
                  <tr className="bg-blue-600 text-white">
                    <th className="px-4 py-3 text-left font-semibold">#</th>
                    <th className="px-4 py-3 text-left font-semibold">Article</th>
                    <th className="px-4 py-3 text-left font-semibold">Category</th>
                    <th className="px-4 py-3 text-left font-semibold">Focus</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  <tr className="hover:bg-blue-50 transition-colors">
                    <td className="px-4 py-3 font-mono text-sm text-blue-600">01</td>
                    <td className="px-4 py-3">
                      <Link href="/post/agentic-rag-for-financial-memo-generation" className="text-blue-600 hover:text-blue-800 font-medium">
                        The USD 156 Million Question: Architecting an Agentic RAG System
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">Business Objective</td>
                    <td className="px-4 py-3 text-sm text-gray-600">ROI & Business Case</td>
                  </tr>
                  <tr className="hover:bg-blue-50 transition-colors">
                    <td className="px-4 py-3 font-mono text-sm text-blue-600">02</td>
                    <td className="px-4 py-3">
                      <Link href="/post/tinyrag-user-journey-mvp-workflow" className="text-blue-600 hover:text-blue-800 font-medium">
                        TinyRAG User Journey: From Login to AI-Powered Insights
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">Business Objective</td>
                    <td className="px-4 py-3 text-sm text-gray-600">UX & User Workflow</td>
                  </tr>
                  <tr className="hover:bg-blue-50 transition-colors">
                    <td className="px-4 py-3 font-mono text-sm text-blue-600">03</td>
                    <td className="px-4 py-3">
                      <Link href="/post/rag-ingestion-pipeline-for-financial-documents" className="text-blue-600 hover:text-blue-800 font-medium">
                        Building a Production-Ready, Asynchronous Ingestion Pipeline
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">Ingestion</td>
                    <td className="px-4 py-3 text-sm text-gray-600">Data Pipeline & Processing</td>
                  </tr>
                  <tr className="hover:bg-blue-50 transition-colors">
                    <td className="px-4 py-3 font-mono text-sm text-blue-600">04</td>
                    <td className="px-4 py-3">
                      <Link href="/post/tinyrag-engineering-deep-dive-qa" className="text-blue-600 hover:text-blue-800 font-medium">
                        TinyRAG Ingestion Deep Dive: Your Questions Answered
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">Ingestion</td>
                    <td className="px-4 py-3 text-sm text-gray-600">Engineering Q&A</td>
                  </tr>
                  <tr className="hover:bg-blue-50 transition-colors">
                    <td className="px-4 py-3 font-mono text-sm text-blue-600">05</td>
                    <td className="px-4 py-3">
                      <Link href="/post/architecting-multi-model-rag-for-global-finance" className="text-blue-600 hover:text-blue-800 font-medium">
                        Architecting Cross-Model Consistency: Model Selection & Prompt Engineering
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">Prompt Tuning</td>
                    <td className="px-4 py-3 text-sm text-gray-600">Multi-Model Strategy</td>
                  </tr>
                  <tr className="hover:bg-blue-50 transition-colors">
                    <td className="px-4 py-3 font-mono text-sm text-blue-600">06</td>
                    <td className="px-4 py-3">
                      <Link href="/post/architecting-production-ready-financial-rag-system" className="text-blue-600 hover:text-blue-800 font-medium">
                        Architecting TinyRAG: A Production-Ready Financial RAG System
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">Engineering Architecture</td>
                    <td className="px-4 py-3 text-sm text-gray-600">System Design & Scalability</td>
                  </tr>
                  <tr className="hover:bg-blue-50 transition-colors">
                    <td className="px-4 py-3 font-mono text-sm text-blue-600">07</td>
                    <td className="px-4 py-3">
                      <Link href="/post/retrieval-strategies-in-financial-rag" className="text-blue-600 hover:text-blue-800 font-medium">
                        Retrieval Strategies: From Dense to Hybrid Approaches
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">Retrieval</td>
                    <td className="px-4 py-3 text-sm text-gray-600">Search & Ranking</td>
                  </tr>
                  <tr className="hover:bg-blue-50 transition-colors">
                    <td className="px-4 py-3 font-mono text-sm text-blue-600">08</td>
                    <td className="px-4 py-3">
                      <Link href="/post/framework-rigorous-evaluation-agentic-postprocessing-financial-rag" className="text-blue-600 hover:text-blue-800 font-medium">
                        A Framework for Rigorous Evaluation and Agentic Post-Processing
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">Evaluation</td>
                    <td className="px-4 py-3 text-sm text-gray-600">Quality Metrics & Testing</td>
                  </tr>
                  <tr className="hover:bg-blue-50 transition-colors">
                    <td className="px-4 py-3 font-mono text-sm text-blue-600">09</td>
                    <td className="px-4 py-3">
                      <Link href="/post/agentic-post-processing-tinyrag" className="text-blue-600 hover:text-blue-800 font-medium">
                        Agentic Post-Processing: Conditional, State-Driven Workflows
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">Agentic Workflow</td>
                    <td className="px-4 py-3 text-sm text-gray-600">Autonomous Processing</td>
                  </tr>
                </tbody>
              </table>
            </div>
            
            <div className="mt-4 p-4 bg-blue-100 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>💡 Learning Tip:</strong> Follow this sequence for optimal understanding. Each article builds upon previous concepts and prepares you for the next challenge in production RAG development.
              </p>
            </div>
          </div>
          
          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <div className="bg-blue-50 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-blue-600">{posts.length}</div>
              <div className="text-sm text-blue-700">Articles</div>
            </div>
            <div className="bg-blue-50 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-blue-600">{RAG_CATEGORIES.length - 1}</div>
              <div className="text-sm text-blue-700">Categories</div>
            </div>
            <div className="bg-blue-50 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-blue-600">100%</div>
              <div className="text-sm text-blue-700">Practical</div>
            </div>
            <div className="bg-blue-50 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-blue-600">Real</div>
              <div className="text-sm text-blue-700">Implementation</div>
            </div>
          </div>
        </section>

        {/* Category Filter */}
        <CategoryFilter 
          selectedCategory={selectedCategory}
          onCategoryChange={setSelectedCategory}
          availableCategories={RAG_CATEGORIES}
        />

        {/* Blog Posts */}
        <section className="space-y-0">
          {filteredPosts.length === 0 ? (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">No Posts Found</h3>
              <p className="text-gray-600">
                {selectedCategory === 'All' 
                  ? 'No RAG implementation posts found. Check back soon for new content!' 
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