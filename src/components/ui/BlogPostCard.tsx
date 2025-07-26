import Link from 'next/link';
import { BlogPost } from '@/types';
import { formatDate } from '@/lib/utils';

interface BlogPostCardProps {
  post: BlogPost;
}

export default function BlogPostCard({ post }: BlogPostCardProps) {
  return (
    <article className="border-b border-gray-200 pb-6 mb-6 last:border-b-0">
      {/* Post Metadata */}
      <div className="flex items-center space-x-3 text-sm text-gray-600 mb-2">
        <span className="bg-gray-100 px-3 py-1 rounded-full font-medium">
          {post.category}
        </span>
        <span>{formatDate(post.date)}</span>
      </div>

      {/* Post Title */}
      <h2 className="text-2xl font-bold mb-3">
        <Link 
          href={`/post/${post.slug}`}
          className="text-gray-900 hover:text-gray-700 transition-colors"
        >
          {post.title}
        </Link>
      </h2>

      {/* Post Summary */}
      {post.summary && (
        <p className="text-gray-700 leading-relaxed">
          {post.summary}
        </p>
      )}
    </article>
  );
} 