import { BlogPost, BlogCategory } from '@/types';
import { marked } from 'marked';

export class ClientBlogService {
  /**
   * Get all posts from API with localStorage fallback
   */
  static async getAllPosts(): Promise<BlogPost[]> {
    try {
      const response = await fetch('/api/posts');
      if (!response.ok) {
        throw new Error('Failed to fetch posts');
      }
      const posts = await response.json();
      return posts.map((post: any) => ({
        ...post,
        date: new Date(post.date),
      }));
    } catch (error) {
      console.error('Error fetching posts from API:', error);
      // Fallback to localStorage
      return this.getLocalStoragePosts();
    }
  }

  /**
   * Get posts by category from API with localStorage fallback
   */
  static async getPostsByCategory(category: BlogCategory): Promise<BlogPost[]> {
    try {
      const response = await fetch(`/api/posts?category=${encodeURIComponent(category)}`);
      if (!response.ok) {
        throw new Error('Failed to fetch posts');
      }
      const posts = await response.json();
      return posts.map((post: any) => ({
        ...post,
        date: new Date(post.date),
      }));
    } catch (error) {
      console.error('Error fetching posts from API:', error);
      // Fallback to localStorage
      const allPosts = this.getLocalStoragePosts();
      if (category === 'All') {
        return allPosts;
      }
      return allPosts.filter(post => post.category === category);
    }
  }

  /**
   * Get post by slug from API with localStorage fallback
   */
  static async getPostBySlug(slug: string): Promise<BlogPost | null> {
    try {
      const response = await fetch(`/api/posts/${encodeURIComponent(slug)}`);
      if (!response.ok) {
        if (response.status === 404) {
          return null;
        }
        throw new Error('Failed to fetch post');
      }
      const post = await response.json();
      return {
        ...post,
        date: new Date(post.date),
      };
    } catch (error) {
      console.error('Error fetching post from API:', error);
      // Fallback to localStorage
      const posts = this.getLocalStoragePosts();
      return posts.find(post => post.slug === slug) || null;
    }
  }

  /**
   * Convert markdown to HTML
   */
  static async markdownToHtml(markdown: string): Promise<string> {
    try {
      return await marked(markdown);
    } catch (error) {
      console.error('Error converting markdown:', error);
      return markdown;
    }
  }

  /**
   * Fallback to localStorage posts
   */
  private static getLocalStoragePosts(): BlogPost[] {
    try {
      const posts = localStorage.getItem('blog-posts');
      if (!posts) return [];
      
      const parsedPosts = JSON.parse(posts);
      return parsedPosts.map((post: BlogPost & { date: string }) => ({
        ...post,
        date: new Date(post.date),
      }));
    } catch (error) {
      console.error('Error loading posts from localStorage:', error);
      return [];
    }
  }

  /**
   * Save post to localStorage (for upload functionality)
   */
  static async savePost(title: string, content: string, category: Exclude<BlogCategory, 'All'>): Promise<BlogPost> {
    const posts = this.getLocalStoragePosts();
    
    const newPost: BlogPost = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      title,
      content,
      category,
      date: new Date(),
      slug: title.toLowerCase().replace(/[^a-z0-9 ]/g, '').replace(/\s+/g, '-'),
      summary: this.extractSummary(content),
    };

    const updatedPosts = [newPost, ...posts];
    
    try {
      localStorage.setItem('blog-posts', JSON.stringify(updatedPosts));
      return newPost;
    } catch (error) {
      console.error('Error saving post:', error);
      throw new Error('Failed to save post');
    }
  }

  /**
   * Extract summary from markdown content
   */
  private static extractSummary(content: string, maxLength: number = 150): string {
    // Remove markdown syntax and get first paragraph
    const text = content
      .replace(/#{1,6}\s+/g, '') // Remove headers
      .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
      .replace(/\*(.*?)\*/g, '$1') // Remove italic
      .replace(/`(.*?)`/g, '$1') // Remove inline code
      .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1') // Remove links
      .split('\n\n')[0] // Get first paragraph
      .trim();

    if (text.length <= maxLength) {
      return text;
    }

    return text.substring(0, maxLength).trim() + '...';
  }
} 