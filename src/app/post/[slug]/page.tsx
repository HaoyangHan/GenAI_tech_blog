'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import Header from '@/components/layout/Header';
import { BlogPost } from '@/types';
import { BlogService } from '@/lib/blog-service';
import { formatDate } from '@/lib/utils';

export default function BlogPostPage() {
  const params = useParams();
  const [post, setPost] = useState<BlogPost | null>(null);
  const [htmlContent, setHtmlContent] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadPost = async () => {
      if (!params.slug || typeof params.slug !== 'string') {
        setLoading(false);
        return;
      }

      const foundPost = BlogService.getPostBySlug(params.slug);
      
      if (foundPost) {
        setPost(foundPost);
        const html = await BlogService.markdownToHtml(foundPost.content);
        setHtmlContent(html);
      }
      
      setLoading(false);
    };

    loadPost();
  }, [params.slug]);

  if (loading) {
    return (
      <div className="min-h-screen bg-white">
        <Header />
        <main className="max-w-4xl mx-auto px-6 py-12">
          <div className="animate-pulse">
            <div className="h-8 bg-gray-200 rounded mb-4 w-1/4"></div>
            <div className="h-12 bg-gray-200 rounded mb-8 w-3/4"></div>
            <div className="space-y-4">
              <div className="h-4 bg-gray-200 rounded w-full"></div>
              <div className="h-4 bg-gray-200 rounded w-5/6"></div>
              <div className="h-4 bg-gray-200 rounded w-4/5"></div>
            </div>
          </div>
        </main>
      </div>
    );
  }

  if (!post) {
    return (
      <div className="min-h-screen bg-white">
        <Header />
        <main className="max-w-4xl mx-auto px-6 py-12 text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Post Not Found</h1>
          <p className="text-gray-600 mb-8">The blog post you're looking for doesn't exist.</p>
          <Link 
            href="/"
            className="inline-flex items-center space-x-2 text-gray-900 hover:text-gray-700 font-medium"
          >
            <ArrowLeft size={16} />
            <span>Back to Articles</span>
          </Link>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      <main className="max-w-4xl mx-auto px-6 py-12">
        {/* Back Navigation */}
        <div className="mb-8">
          <Link 
            href="/"
            className="inline-flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors font-medium"
          >
            <ArrowLeft size={16} />
            <span>Back to Articles</span>
          </Link>
        </div>

        {/* Post Metadata */}
        <div className="flex items-center space-x-3 text-sm text-gray-600 mb-6">
          <span className="bg-gray-100 px-3 py-1 rounded-full font-medium">
            {post.category}
          </span>
          <span>{formatDate(post.date)}</span>
        </div>

        {/* Post Title */}
        <h1 className="text-5xl font-bold text-gray-900 mb-12 leading-tight">
          {post.title}
        </h1>

        {/* Post Content */}
        <article 
          className="prose-custom"
          dangerouslySetInnerHTML={{ __html: htmlContent }}
        />
      </main>
    </div>
  );
} 