import { NextResponse } from 'next/server';
import { FileBlogService } from '@/lib/file-blog-service';

export async function GET(
  request: Request,
  { params }: { params: { slug: string } }
) {
  try {
    const post = await FileBlogService.getPostBySlug(params.slug);
    
    if (!post) {
      return NextResponse.json({ error: 'Post not found' }, { status: 404 });
    }

    return NextResponse.json(post);
  } catch (error) {
    console.error('Error fetching post:', error);
    return NextResponse.json({ error: 'Failed to fetch post' }, { status: 500 });
  }
} 