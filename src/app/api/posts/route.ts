import { NextRequest, NextResponse } from 'next/server';
import { FileBlogService } from '@/lib/file-blog-service';
import { KnowledgeBase } from '@/types';

// Force dynamic rendering for this API route
export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const category = searchParams.get('category');
    const knowledgeBase = searchParams.get('knowledgeBase') as KnowledgeBase;

    let posts;
    
    if (knowledgeBase) {
      // Get posts from specific knowledge base
      if (category && category !== 'All') {
        posts = await FileBlogService.getPostsByCategoryAndKnowledgeBase(category as any, knowledgeBase);
      } else {
        posts = await FileBlogService.getPostsByKnowledgeBase(knowledgeBase);
      }
    } else {
      // Get all posts or filter by category across all knowledge bases
      if (category && category !== 'All') {
        posts = await FileBlogService.getPostsByCategory(category as any);
      } else {
        posts = await FileBlogService.getAllPosts();
      }
    }

    return NextResponse.json(posts);
  } catch (error) {
    console.error('Error fetching posts:', error);
    return NextResponse.json({ error: 'Failed to fetch posts' }, { status: 500 });
  }
} 