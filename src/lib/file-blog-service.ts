import { BlogPost, BlogCategory, KnowledgeBase, RAG_CATEGORIES, FOUNDATION_CATEGORIES } from '@/types';
import { formatDate } from './utils';
import matter from 'gray-matter';
import { marked } from 'marked';
import hljs from 'highlight.js';
import katex from 'katex';
import fs from 'fs';
import path from 'path';

export class FileBlogService {
  private static postsDirectory = path.join(process.cwd(), 'posts');
  private static foundationPostsDirectory = path.join(process.cwd(), 'llm_foundation_posts');

  /**
   * Parse filename to extract a clean title (converts underscores to dots in numeric patterns)
   */
  private static parseFilenameTitle(filename: string): string {
    const nameWithoutExt = filename.replace('.md', '');
    
    // Convert patterns like "1_1_foundations" to "1.1.foundations"
    const converted = nameWithoutExt.replace(/(\d+)_(\d+)/g, '$1.$2');
    
    // Convert remaining underscores and hyphens to spaces and title case
    return converted
      .replace(/[-_]/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }

  /**
   * Get all blog posts from both knowledge bases
   */
  static async getAllPosts(): Promise<BlogPost[]> {
    // Return empty array if running on client side
    if (typeof window !== 'undefined') {
      console.warn('FileBlogService should only be used on the server side');
      return [];
    }

    try {
      const ragPosts = await this.getPostsByKnowledgeBase('rag');
      const foundationPosts = await this.getPostsByKnowledgeBase('foundations');
      
      const allPosts = [...ragPosts, ...foundationPosts];
      
      // Sort posts by date (newest first)
      return allPosts.sort((a, b) => b.date.getTime() - a.date.getTime());
    } catch (error) {
      console.error('Error loading all posts:', error);
      return [];
    }
  }

  /**
   * Get posts from a specific knowledge base
   */
  static async getPostsByKnowledgeBase(knowledgeBase: KnowledgeBase): Promise<BlogPost[]> {
    if (typeof window !== 'undefined') {
      console.warn('FileBlogService should only be used on the server side');
      return [];
    }

    try {
      if (knowledgeBase === 'rag') {
        return await this.getRAGPosts();
      } else if (knowledgeBase === 'foundations') {
        return await this.getFoundationPosts();
      }
      return [];
    } catch (error) {
      console.error(`Error loading posts from knowledge base ${knowledgeBase}:`, error);
      return [];
    }
  }

  /**
   * Get RAG implementation posts from the posts directory
   */
  private static async getRAGPosts(): Promise<BlogPost[]> {
    // Check if posts directory exists
    if (!fs.existsSync(this.postsDirectory)) {
      console.log('Posts directory not found, creating it...');
      fs.mkdirSync(this.postsDirectory, { recursive: true });
      return [];
    }

    const validCategories = ['Business Objective', 'Engineering Architecture', 'Ingestion', 'Retrieval', 'Generation', 'Evaluation', 'Prompt Tuning', 'Agentic Workflow'];
    const allPosts: BlogPost[] = [];

    // Read from category folders
    for (const category of validCategories) {
      const categoryPath = path.join(this.postsDirectory, category);
      
      if (fs.existsSync(categoryPath) && fs.statSync(categoryPath).isDirectory()) {
        const categoryFiles = fs.readdirSync(categoryPath);
        const markdownFiles = categoryFiles.filter(name => name.endsWith('.md'));

        const categoryPosts = await Promise.all(
          markdownFiles.map(async (fileName) => {
            const filePath = path.join(categoryPath, fileName);
            const fileContents = fs.readFileSync(filePath, 'utf8');
            
            // Parse front matter
            const { data, content } = matter(fileContents);
            
            // Extract metadata with defaults and validation
            const slug = data.slug || fileName.replace('.md', '');
            const title = data.title || this.extractTitleFromContent(content) || this.parseFilenameTitle(fileName);
            
            const date = data.date ? new Date(data.date) : new Date();
            const summary = data.summary || this.extractSummary(content);
            
            // Process markdown content to HTML with LaTeX support
            const htmlContent = await this.markdownToHtml(content);
            
            return {
              id: slug,
              title,
              content: htmlContent,
              category: category as BlogCategory,
              date,
              slug,
              summary,
              tags: data.tags || [],
              author: data.author || 'Haoyang Han',
              knowledgeBase: 'rag' as KnowledgeBase,
            } as BlogPost;
          })
        );
        
        allPosts.push(...categoryPosts);
      }
    }

    return allPosts;
  }

  /**
   * Get statistical foundation posts from the llm_foundation_posts directory
   */
  private static async getFoundationPosts(): Promise<BlogPost[]> {
    // Check if foundation posts directory exists
    if (!fs.existsSync(this.foundationPostsDirectory)) {
      console.log('Foundation posts directory not found');
      return [];
    }

    const allPosts: BlogPost[] = [];

    // Recursively find all markdown files in the foundation directory
    const findMarkdownFiles = (dir: string): string[] => {
      const files: string[] = [];
      const items = fs.readdirSync(dir);
      
      for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);
        
        if (stat.isDirectory()) {
          files.push(...findMarkdownFiles(fullPath));
        } else if (item.endsWith('.md')) {
          files.push(fullPath);
        }
      }
      
      return files;
    };

    const markdownFiles = findMarkdownFiles(this.foundationPostsDirectory);

    const foundationPosts = await Promise.all(
      markdownFiles.map(async (filePath) => {
        const fileContents = fs.readFileSync(filePath, 'utf8');
        
        // Parse front matter
        const { data, content } = matter(fileContents);
        
        // Extract metadata with defaults and validation
        const fileName = path.basename(filePath);
        const slug = data.slug || fileName.replace('.md', '');
        const title = data.title || this.extractTitleFromContent(content) || this.parseFilenameTitle(fileName);
        
        const date = data.date ? new Date(data.date) : new Date();
        const summary = data.summary || this.extractSummary(content);
        
        // Process markdown content to HTML with LaTeX support
        const htmlContent = await this.markdownToHtml(content);
        
        return {
          id: slug,
          title,
          content: htmlContent,
          category: (data.category as BlogCategory) || 'Statistical Deep Dive',
          date,
          slug,
          summary,
          tags: data.tags || [],
          author: data.author || 'Haoyang Han',
          knowledgeBase: 'foundations' as KnowledgeBase,
        } as BlogPost;
      })
    );
    
    allPosts.push(...foundationPosts);

    return allPosts;
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
   * Get posts filtered by category and knowledge base
   */
  static async getPostsByCategoryAndKnowledgeBase(category: BlogCategory, knowledgeBase: KnowledgeBase): Promise<BlogPost[]> {
    const posts = await this.getPostsByKnowledgeBase(knowledgeBase);
    
    if (category === 'All') {
      return posts;
    }
    
    return posts.filter(post => post.category === category);
  }

  /**
   * Get a single blog post by slug
   */
  static async getPostBySlug(slug: string): Promise<BlogPost | null> {
    try {
      const posts = await this.getAllPosts();
      return posts.find(post => post.slug === slug) || null;
    } catch (error) {
      console.error('Error getting post by slug:', slug, error);
      return null;
    }
  }

  /**
   * Convert markdown content to HTML with enhanced Python syntax highlighting and LaTeX math support
   */
  static async markdownToHtml(markdown: string): Promise<string> {
    try {
      // First, process LaTeX equations
      const processedMarkdown = this.processLaTeXEquations(markdown);
      
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
          console.warn('LaTeX display math rendering failed:', err);
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
            console.warn('LaTeX inline math rendering failed:', err);
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
   * Extract title from markdown content (first heading)
   */
  private static extractTitleFromContent(content: string): string | null {
    // Look for the first heading (# Title)
    const headingMatch = content.match(/^#\s+(.+)$/m);
    if (headingMatch) {
      return headingMatch[1].trim();
    }
    return null;
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

 