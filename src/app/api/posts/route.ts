import { NextResponse } from 'next/server';
import { FileBlogService } from '@/lib/file-blog-service';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const category = searchParams.get('category');

    let posts;
    if (category && category !== 'All') {
      posts = await FileBlogService.getPostsByCategory(category as any);
    } else {
      posts = await FileBlogService.getAllPosts();
    }

    return NextResponse.json(posts);
  } catch (error) {
    console.error('Error fetching posts:', error);
    return NextResponse.json({ error: 'Failed to fetch posts' }, { status: 500 });
  }
} 