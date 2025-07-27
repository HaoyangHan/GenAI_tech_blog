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
          <p className="text-lg text-gray-600 mb-6">
            Complete journey from business objectives to production deployment of Retrieval-Augmented Generation systems. 
            Explore real-world implementation details, engineering decisions, and performance optimizations.
          </p>
          
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