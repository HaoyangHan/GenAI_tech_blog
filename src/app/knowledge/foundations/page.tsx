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
          <p className="text-lg text-gray-600 mb-6">
            Essential mathematical and statistical foundations for modern data science and machine learning. 
            Deep dive into the theoretical underpinnings of LLMs, traditional ML algorithms, probability theory, 
            and advanced optimization techniques with comprehensive proofs and implementations.
          </p>
          
          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <div className="bg-purple-50 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-purple-600">{posts.length}</div>
              <div className="text-sm text-purple-700">Articles</div>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-purple-600">∞</div>
              <div className="text-sm text-purple-700">Math Proofs</div>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-purple-600">100%</div>
              <div className="text-sm text-purple-700">Rigorous</div>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-purple-600">Deep</div>
              <div className="text-sm text-purple-700">Theory</div>
            </div>
          </div>

          {/* Topics Overview */}
          <div className="bg-purple-50 rounded-lg p-6 mb-8">
            <h3 className="text-lg font-semibold text-purple-900 mb-4">Topics Covered</h3>
            <div className="grid md:grid-cols-2 gap-4 text-sm text-purple-800">
              <div>
                <h4 className="font-medium mb-2">LLM Base Knowledge</h4>
                <ul className="space-y-1 text-purple-700">
                  <li>• Attention Mechanisms & Transformers</li>
                  <li>• Tokenization & Language Models</li>
                  <li>• Fine-tuning & Optimization</li>
                  <li>• RAG Theory & Evaluation</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">Traditional ML</h4>
                <ul className="space-y-1 text-purple-700">
                  <li>• Probability & Statistics</li>
                  <li>• Evaluation Metrics</li>
                  <li>• Data Processing with Pandas</li>
                  <li>• Scikit-learn Fundamentals</li>
                </ul>
              </div>
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