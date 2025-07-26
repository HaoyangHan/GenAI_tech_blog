import { BlogPost, BlogCategory } from '@/types';
import { generateId, titleToSlug, extractSummary } from './utils';
import { marked } from 'marked';

const STORAGE_KEY = 'blog-posts';

export class BlogService {
  /**
   * Get all blog posts from localStorage
   */
  static getAllPosts(): BlogPost[] {
    if (typeof window === 'undefined') return [];
    
    try {
      const posts = localStorage.getItem(STORAGE_KEY);
      if (!posts) return [];
      
      const parsedPosts = JSON.parse(posts);
      return parsedPosts.map((post: any) => ({
        ...post,
        date: new Date(post.date),
      }));
    } catch (error) {
      console.error('Error loading posts:', error);
      return [];
    }
  }

  /**
   * Get posts filtered by category
   */
  static getPostsByCategory(category: BlogCategory): BlogPost[] {
    const allPosts = this.getAllPosts();
    
    if (category === 'All') {
      return allPosts;
    }
    
    return allPosts.filter(post => post.category === category);
  }

  /**
   * Get a single post by slug
   */
  static getPostBySlug(slug: string): BlogPost | null {
    const posts = this.getAllPosts();
    return posts.find(post => post.slug === slug) || null;
  }

  /**
   * Save a new blog post
   */
  static async savePost(title: string, content: string, category: Exclude<BlogCategory, 'All'>): Promise<BlogPost> {
    const posts = this.getAllPosts();
    
    const newPost: BlogPost = {
      id: generateId(),
      title,
      content,
      category,
      date: new Date(),
      slug: titleToSlug(title),
      summary: extractSummary(content),
    };

    const updatedPosts = [newPost, ...posts];
    
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedPosts));
      return newPost;
    } catch (error) {
      console.error('Error saving post:', error);
      throw new Error('Failed to save post');
    }
  }

  /**
   * Delete a post by ID
   */
  static deletePost(id: string): boolean {
    try {
      const posts = this.getAllPosts();
      const updatedPosts = posts.filter(post => post.id !== id);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedPosts));
      return true;
    } catch (error) {
      console.error('Error deleting post:', error);
      return false;
    }
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
   * Initialize with sample data if no posts exist
   */
  static initializeSampleData(): void {
    const existingPosts = this.getAllPosts();
    
    if (existingPosts.length === 0) {
      // Sample content from the provided markdown file
      const sampleContent = `# 1.1 LLM Base Knowledge

> This guide provides a deep dive into foundational concepts for modern data science and deep learning interviews, covering the Attention Mechanism, advanced Optimization Algorithms, Regularization Techniques, and common Loss Functions. It begins with a thorough explanation of the theory behind each topic, complete with mathematical formulations and proofs. This is followed by a curated set of theoretical and practical interview questions, including full Python and PyTorch code implementations, to solidify understanding and prepare for real-world interview scenarios.

## Knowledge Section

This section breaks down the core theoretical knowledge required to understand advanced machine learning models. We will cover the mechanics of attention, the evolution of optimization algorithms, methods to prevent overfitting, and the mathematical basis for classification loss functions.

*(Note: The information herein is current as of my last update in early 2023. While the core principles are timeless, newer model variants or optimizers may have emerged since.)*

### The Attention Mechanism

The Attention Mechanism, originally proposed for machine translation in Bahdanau et al. (2014) and refined in Vaswani et al.'s "Attention Is All You Need" (2017), has become a cornerstone of modern deep learning, particularly in Natural Language Processing (NLP). It allows a model to dynamically focus on the most relevant parts of the input sequence when producing an output, overcoming the fixed-length context vector bottleneck of traditional sequence-to-sequence models like RNNs.

#### The Core Idea: Query, Key, and Value

At its heart, attention can be described as a process of mapping a **Query (Q)** and a set of **Key-Value (K-V)** pairs to an output.

*   **Query (Q):** Represents the current context or the element we are trying to compute an output for. For example, in a decoder, it could be the representation of the previously generated word.
*   **Keys (K):** A set of vectors representing different parts of the input sequence. Each key is associated with a value.
*   **Values (V):** A set of vectors that contain the actual information from the input sequence. Each value corresponds to a key.

The process involves three main steps:
1.  **Calculate Attention Scores:** A compatibility function is used to compute a score between the Query and each Key. A higher score means the corresponding Value is more relevant to the Query. The most common compatibility function is the dot product.
2.  **Normalize Scores to Weights:** The raw scores are passed through a \`softmax\` function to convert them into a probability distribution. These normalized scores are the **attention weights**, and they sum to 1.
3.  **Compute Weighted Sum:** The attention weights are multiplied by their corresponding Value vectors, and the results are summed up to produce the final output vector. This output is a weighted representation of the input values, where the weights are determined by the query's relevance to each key.

## Practical Implementation

Here's how you can implement attention in PyTorch:

\`\`\`python
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, v)
        return output, attention_weights
\`\`\`

This implementation demonstrates the core concepts of attention mechanisms and their practical application in deep learning models.`;

      this.savePost(
        'LLM Base Knowledge - Foundations',
        sampleContent,
        'Engineering Architecture'
      );
    }
  }
} 