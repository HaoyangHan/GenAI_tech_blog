'use client';

import { useState, useEffect } from 'react';
import Header from '@/components/layout/Header';
import CategoryFilter from '@/components/ui/CategoryFilter';
import BlogPostCard from '@/components/ui/BlogPostCard';
import { BlogPost, BlogCategory } from '@/types';
import { BlogService } from '@/lib/blog-service';

export default function HomePage() {
  const [posts, setPosts] = useState<BlogPost[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<BlogCategory>('All');
  const [filteredPosts, setFilteredPosts] = useState<BlogPost[]>([]);

  useEffect(() => {
    // Initialize sample data and load posts
    BlogService.initializeSampleData();
    const allPosts = BlogService.getAllPosts();
    setPosts(allPosts);
  }, []);

  useEffect(() => {
    // Filter posts when category changes
    const filtered = BlogService.getPostsByCategory(selectedCategory);
    setFilteredPosts(filtered);
  }, [selectedCategory, posts]);

  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      <main className="max-w-4xl mx-auto px-6 py-12">
        {/* Page Title */}
        <section className="mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            RAG Implementation Details
          </h1>
          <p className="text-lg text-gray-600">
            Exploring the complete journey from business objectives to production deployment
          </p>
        </section>

        {/* Category Filter */}
        <CategoryFilter 
          selectedCategory={selectedCategory}
          onCategoryChange={setSelectedCategory}
        />

        {/* Blog Posts */}
        <section className="space-y-0">
          {filteredPosts.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-600 text-lg">
                {selectedCategory === 'All' 
                  ? 'No blog posts found. Upload your first post!' 
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