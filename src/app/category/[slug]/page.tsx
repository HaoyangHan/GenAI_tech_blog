'use client';

import { useState, useEffect } from 'react';
import { notFound } from 'next/navigation';
import Header from '@/components/layout/Header';
import BlogPostCard from '@/components/ui/BlogPostCard';
import { BlogPost, BlogCategory, BLOG_CATEGORIES } from '@/types';
import { ClientBlogService } from '@/lib/client-blog-service';
import { Tag, BookOpen, Filter } from 'lucide-react';

interface CategoryPageProps {
  params: {
    slug: string;
  };
}

export default function CategoryPage({ params }: CategoryPageProps) {
  const [posts, setPosts] = useState<BlogPost[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Convert slug to category name
  const categoryName = decodeURIComponent(params.slug)
    .split('-')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');

  // Validate if it's a valid category
  const isValidCategory = BLOG_CATEGORIES.includes(categoryName as BlogCategory);

  useEffect(() => {
    const loadCategoryPosts = async () => {
      if (!isValidCategory) {
        setError('Invalid category');
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        const categoryPosts = await ClientBlogService.getPostsByCategory(categoryName as BlogCategory);
        setPosts(categoryPosts);
      } catch (err) {
        console.error('Error loading category posts:', err);
        setError('Failed to load posts');
      } finally {
        setLoading(false);
      }
    };

    loadCategoryPosts();
  }, [categoryName, isValidCategory]);

  if (!isValidCategory) {
    return notFound();
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-white">
        <Header />
        <main className="max-w-6xl mx-auto px-6 py-12">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto"></div>
            <p className="mt-4 text-gray-600">Loading posts...</p>
          </div>
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-white">
        <Header />
        <main className="max-w-6xl mx-auto px-6 py-12">
          <div className="text-center">
            <p className="text-red-600">{error}</p>
          </div>
        </main>
      </div>
    );
  }

  // Get category description
  const getCategoryDescription = (category: string): string => {
    const descriptions: Record<string, string> = {
      'Business Objective': 'Strategic business goals, ROI considerations, and organizational impact of GenAI initiatives.',
      'Engineering Architecture': 'System design, infrastructure patterns, and technical architecture for scalable AI solutions.',
      'Ingestion': 'Data pipeline design, ETL processes, and knowledge base construction for AI systems.',
      'Retrieval': 'Vector search, semantic retrieval, and information retrieval techniques for RAG systems.',
      'Generation': 'Text generation, model fine-tuning, and output optimization strategies.',
      'Evaluation': 'Metrics, benchmarking, and assessment frameworks for AI system performance.',
      'Prompt Tuning': 'Prompt engineering, optimization techniques, and effective communication with LLMs.',
      'Agentic Workflow': 'Multi-agent systems, workflow orchestration, and autonomous AI agents.',
      'GenAI Knowledge': 'Fundamental concepts, mathematical foundations, and core principles of Generative AI.'
    };
    return descriptions[category] || 'Explore posts in this category.';
  };

  // Get category icon
  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Business Objective':
        return <Tag className="w-6 h-6" />;
      case 'Engineering Architecture':
        return <BookOpen className="w-6 h-6" />;
      case 'GenAI Knowledge':
        return <Filter className="w-6 h-6" />;
      default:
        return <BookOpen className="w-6 h-6" />;
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      <main className="max-w-6xl mx-auto px-6 py-12">
        {/* Category Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4">
            {getCategoryIcon(categoryName)}
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            {categoryName}
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            {getCategoryDescription(categoryName)}
          </p>
          <div className="mt-6 inline-flex items-center space-x-2 text-sm text-gray-500">
            <Filter className="w-4 h-4" />
            <span>{posts.length} {posts.length === 1 ? 'post' : 'posts'}</span>
          </div>
        </div>

        {/* Posts Grid */}
        {posts.length > 0 ? (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {posts.map((post) => (
              <BlogPostCard key={post.id} post={post} />
            ))}
          </div>
        ) : (
          <div className="text-center py-16">
            <BookOpen className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-900 mb-2">No posts yet</h3>
            <p className="text-gray-600">
              There are no posts in the "{categoryName}" category yet. Check back soon for new content!
            </p>
          </div>
        )}

        {/* Category Navigation */}
        <div className="mt-16 pt-12 border-t border-gray-200">
          <h2 className="text-2xl font-bold text-gray-900 mb-8 text-center">Explore Other Categories</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {BLOG_CATEGORIES.filter(cat => cat !== 'All' && cat !== categoryName).map((category) => (
              <a
                key={category}
                href={`/category/${category.toLowerCase().replace(/\s+/g, '-')}`}
                className="block p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors text-center"
              >
                <div className="text-sm font-medium text-gray-900">{category}</div>
              </a>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
} 