import { BlogPost, BlogCategory } from '@/types';
import { marked } from 'marked';
import hljs from 'highlight.js';
import katex from 'katex';

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
   * Convert markdown to HTML with enhanced Python syntax highlighting and LaTeX math support
   */
  static async markdownToHtml(markdown: string): Promise<string> {
    try {
      // First, process LaTeX equations
      let processedMarkdown = this.processLaTeXEquations(markdown);
      
      // Create a custom renderer for code blocks
      const renderer = new marked.Renderer();
      
      // Override the code rendering to use highlight.js with Python as default
      renderer.code = function(code: string, language?: string) {
        // Default to Python if no language specified
        const lang = language || 'python';
        
        let highlightedCode = code;
        
        // Try to highlight with the specified language
        if (hljs.getLanguage(lang)) {
          try {
            highlightedCode = hljs.highlight(code, { language: lang }).value;
          } catch (err) {
            console.warn(`Syntax highlighting failed for language: ${lang}`, err);
            // Fallback to Python highlighting
            try {
              highlightedCode = hljs.highlight(code, { language: 'python' }).value;
            } catch (fallbackErr) {
              highlightedCode = code; // Return plain code if both fail
            }
          }
        } else {
          // If language not supported, try Python as fallback
          try {
            highlightedCode = hljs.highlight(code, { language: 'python' }).value;
          } catch (err) {
            highlightedCode = code; // Return plain code if highlighting fails
          }
        }
        
        return `<pre><code class="hljs language-${lang}">${highlightedCode}</code></pre>`;
      };

      return await marked(processedMarkdown, { renderer });
    } catch (error) {
      console.error('Error converting markdown:', error);
      return markdown;
    }
  }

  /**
   * Process LaTeX equations in markdown content
   */
  private static processLaTeXEquations(markdown: string): string {
    try {
      // Normalize line endings first (convert \r\n to \n and handle other variations)
      let processedMarkdown = markdown
        .replace(/\r\n/g, '\n')  // Windows CRLF -> LF
        .replace(/\r/g, '\n')    // Old Mac CR -> LF
        .replace(/\u2028/g, '\n') // Unicode line separator
        .replace(/\u2029/g, '\n'); // Unicode paragraph separator
      
      // STEP 1: First, protect display math by replacing it with placeholders
      const displayMathPlaceholders: string[] = [];
      let displayMathCount = 0;
      
      // Find and store all display math expressions
      processedMarkdown = processedMarkdown.replace(/\$\$\s*([\s\S]*?)\s*\$\$/g, (match, equation) => {
        displayMathCount++;
        const cleanEquation = equation.trim();
        if (!cleanEquation) {
          return match;
        }
        
        try {
          const renderedMath = katex.renderToString(cleanEquation, {
            displayMode: true,
            throwOnError: false,
            strict: false,
            trust: true,
            output: 'html'
          });
          
          const placeholder = `__DISPLAY_MATH_${displayMathCount}__`;
          displayMathPlaceholders.push(`<div class="math-display" data-latex="$$${cleanEquation}$$">${renderedMath}</div>`);
          return placeholder;
        } catch (err) {
          console.error(`Display math ${displayMathCount}: Rendering failed:`, err, 'for equation:', cleanEquation.substring(0, 100));
          const placeholder = `__DISPLAY_MATH_ERROR_${displayMathCount}__`;
          displayMathPlaceholders.push(`<div class="math-error">Display Math Error: ${cleanEquation}</div>`);
          return placeholder;
        }
      });

      // STEP 2: Process inline math ($...$) with better conflict avoidance
      let inlineMathCount = 0;
      
      // Split into blocks to avoid processing math inside code blocks
      const blocks = processedMarkdown.split(/(```[\s\S]*?```|`[^`]*`)/);
      
      const processedBlocks = blocks.map((block, index) => {
        // Skip code blocks (odd indices after split)
        if (index % 2 === 1 || block.startsWith('```') || block.startsWith('`')) {
          return block;
        }
        
        // Process inline math in non-code blocks
        return block.replace(/\$([^$\n\r]+?)\$/g, (match, equation) => {
          inlineMathCount++;
          try {
            const cleanEquation = equation.trim();
            if (!cleanEquation) {
              return match;
            }
            
            // Skip if it looks like it might be a code snippet or contains backticks
            if (cleanEquation.includes('```') || cleanEquation.includes('`')) {
              return match;
            }
            
            const renderedMath = katex.renderToString(cleanEquation, {
              displayMode: false,
              throwOnError: false,
              strict: false,
              trust: true,
              output: 'html'
            });
            
            return `<span class="math-inline" data-latex="$${cleanEquation}$">${renderedMath}</span>`;
          } catch (err) {
            console.error(`Inline math ${inlineMathCount}: Rendering failed:`, err, 'for equation:', equation);
            return `<span class="math-error">Inline Math Error: ${equation}</span>`;
          }
        });
      });

      let result = processedBlocks.join('');

      // STEP 3: Restore display math placeholders
      displayMathPlaceholders.forEach((mathHtml, index) => {
        const placeholder = `__DISPLAY_MATH_${index + 1}__`;
        const errorPlaceholder = `__DISPLAY_MATH_ERROR_${index + 1}__`;
        result = result.replace(placeholder, mathHtml);
        result = result.replace(errorPlaceholder, mathHtml);
      });

      return result;
    } catch (error) {
      console.error('Critical error in LaTeX processing:', error);
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
      tags: [],
      author: 'Haoyang Han',
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