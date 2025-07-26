import { BlogPost, BlogCategory } from '@/types';
import { formatDate } from './utils';
import matter from 'gray-matter';
import { marked } from 'marked';
import fs from 'fs';
import path from 'path';

export class FileBlogService {
  private static postsDirectory = path.join(process.cwd(), 'posts');

  /**
   * Get all blog posts from markdown files
   */
  static async getAllPosts(): Promise<BlogPost[]> {
    // Return empty array if running on client side
    if (typeof window !== 'undefined') {
      console.warn('FileBlogService should only be used on the server side');
      return [];
    }

    try {
      // Check if posts directory exists
      if (!fs.existsSync(this.postsDirectory)) {
        console.log('Posts directory not found, creating it...');
        fs.mkdirSync(this.postsDirectory, { recursive: true });
        return [];
      }

      const fileNames = fs.readdirSync(this.postsDirectory);
      const markdownFiles = fileNames.filter(name => name.endsWith('.md'));

      const posts = await Promise.all(
        markdownFiles.map(async (fileName) => {
          const filePath = path.join(this.postsDirectory, fileName);
          const fileContents = fs.readFileSync(filePath, 'utf8');
          
          // Parse front matter
          const { data, content } = matter(fileContents);
          
          // Extract metadata with defaults
          const slug = data.slug || fileName.replace('.md', '');
          const title = data.title || 'Untitled';
          const category = data.category || 'Engineering Architecture';
          const date = data.date ? new Date(data.date) : new Date();
          const summary = data.summary || this.extractSummary(content);
          
          return {
            id: slug,
            title,
            content,
            category: category as BlogCategory,
            date,
            slug,
            summary,
            tags: data.tags || [],
            author: data.author || 'Anonymous',
          } as BlogPost;
        })
      );

      // Sort posts by date (newest first)
      return posts.sort((a, b) => b.date.getTime() - a.date.getTime());
    } catch (error) {
      console.error('Error loading posts from files:', error);
      return [];
    }
  }

  /**
   * Get posts filtered by category
   */
  static async getPostsByCategory(category: BlogCategory): Promise<BlogPost[]> {
    const allPosts = await this.getAllPosts();
    
    if (category === 'All') {
      return allPosts;
    }
    
    return allPosts.filter(post => post.category === category);
  }

  /**
   * Get a single post by slug
   */
  static async getPostBySlug(slug: string): Promise<BlogPost | null> {
    const posts = await this.getAllPosts();
    return posts.find(post => post.slug === slug) || null;
  }

  /**
   * Convert markdown content to HTML
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

  /**
   * Create a new blog post file
   */
  static async createPost(
    title: string,
    content: string,
    category: Exclude<BlogCategory, 'All'>,
    options: {
      slug?: string;
      summary?: string;
      tags?: string[];
      author?: string;
    } = {}
  ): Promise<string> {
    if (typeof window !== 'undefined') {
      throw new Error('File operations can only be performed on the server side');
    }

    const slug = options.slug || title.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    const fileName = `${slug}.md`;
    const filePath = path.join(this.postsDirectory, fileName);

    // Check if file already exists
    if (fs.existsSync(filePath)) {
      throw new Error(`Post with slug "${slug}" already exists`);
    }

    // Create front matter
    const frontMatter = {
      title,
      category,
      date: new Date().toISOString().split('T')[0], // YYYY-MM-DD format
      summary: options.summary || this.extractSummary(content),
      slug,
      tags: options.tags || [],
      author: options.author || 'GenAI Team',
    };

    // Combine front matter and content
    const fileContent = matter.stringify(content, frontMatter);

    // Write file
    fs.writeFileSync(filePath, fileContent, 'utf8');

    return slug;
  }

  /**
   * List all available markdown files
   */
  static getPostFileNames(): string[] {
    if (typeof window !== 'undefined') {
      return [];
    }

    try {
      if (!fs.existsSync(this.postsDirectory)) {
        return [];
      }

      return fs.readdirSync(this.postsDirectory)
        .filter(name => name.endsWith('.md'))
        .map(name => name.replace('.md', ''));
    } catch (error) {
      console.error('Error listing post files:', error);
      return [];
    }
  }
}

// Export a hybrid service that uses file-based on server, localStorage on client
export class HybridBlogService {
  /**
   * Get all posts - uses file system on server, localStorage on client
   */
  static async getAllPosts(): Promise<BlogPost[]> {
    if (typeof window === 'undefined') {
      // Server side - use file system
      return await FileBlogService.getAllPosts();
    } else {
      // Client side - use localStorage (fallback)
      return this.getLocalStoragePosts();
    }
  }

  /**
   * Get posts by category
   */
  static async getPostsByCategory(category: BlogCategory): Promise<BlogPost[]> {
    const allPosts = await this.getAllPosts();
    
    if (category === 'All') {
      return allPosts;
    }
    
    return allPosts.filter(post => post.category === category);
  }

  /**
   * Get post by slug
   */
  static async getPostBySlug(slug: string): Promise<BlogPost | null> {
    const posts = await this.getAllPosts();
    return posts.find(post => post.slug === slug) || null;
  }

  /**
   * Convert markdown to HTML
   */
  static async markdownToHtml(markdown: string): Promise<string> {
    return await FileBlogService.markdownToHtml(markdown);
  }

  /**
   * Fallback to localStorage posts on client side
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
} 