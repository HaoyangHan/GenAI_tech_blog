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
      return parsedPosts.map((post: BlogPost & { date: string }) => ({
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
2.  **Normalize Scores to Weights:** The raw scores are passed through a softmax function to convert them into a probability distribution. These normalized scores are the **attention weights**, and they sum to 1.
3.  **Compute Weighted Sum:** The attention weights are multiplied by their corresponding Value vectors, and the results are summed up to produce the final output vector. This output is a weighted representation of the input values, where the weights are determined by the query's relevance to each key.

#### Scaled Dot-Product Attention

The Transformer architecture popularized a specific implementation called **Scaled Dot-Product Attention**.

Given a query Q, keys K, and values V, the output is computed as:

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

Let's break down this formula:
*   $Q \\in \\mathbb{R}^{n \\times d_k}$, $K \\in \\mathbb{R}^{m \\times d_k}$, $V \\in \\mathbb{R}^{m \\times d_v}$
    *   $n$: sequence length of queries.
    *   $m$: sequence length of keys/values.
    *   $d_k$: dimension of queries and keys.
    *   $d_v$: dimension of values.

#### Multi-Head Attention

Instead of performing a single attention function with high-dimensional keys, values, and queries, it was found to be more beneficial to linearly project the queries, keys, and values h times (the number of "heads") to different, learned, linear subspaces.

$$\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h)W^O$$

where $\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$.

**Why is this useful?** Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging can dilute important signals. Multiple heads allow each head to specialize and focus on a different aspect of the input.

## Practical Implementation

Here's how you can implement Scaled Dot-Product Attention in PyTorch:

\`\`\`python
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    """
    Implements the Scaled Dot-Product Attention mechanism.
    
    Args:
        dropout (float): Dropout probability.
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Performs the forward pass of the attention mechanism.
        
        Args:
            q (torch.Tensor): Query tensor. Shape: (batch_size, num_heads, seq_len_q, d_k)
            k (torch.Tensor): Key tensor. Shape: (batch_size, num_heads, seq_len_k, d_k)
            v (torch.Tensor): Value tensor. Shape: (batch_size, num_heads, seq_len_v, d_v)
            mask (torch.Tensor, optional): Mask to be applied to attention scores.
        
        Returns:
            torch.Tensor: The output of the attention mechanism
            torch.Tensor: The attention weights
        """
        # Ensure the key dimension matches for dot product
        d_k = k.size(-1)
        
        # 1. Compute dot product scores: (Q * K^T)
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # 2. Scale the scores
        scores = scores / math.sqrt(d_k)
        
        # 3. Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 4. Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply dropout to the attention weights
        attention_weights = self.dropout(attention_weights)
        
        # 5. Compute the weighted sum of values
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights

# Example Usage
if __name__ == '__main__':
    batch_size = 8
    num_heads = 4
    seq_len = 10
    d_k = 16  # Dimension of keys/queries
    d_v = 32  # Dimension of values
    
    # Create random input tensors
    q = torch.randn(batch_size, num_heads, seq_len, d_k)
    k = torch.randn(batch_size, num_heads, seq_len, d_k)
    v = torch.randn(batch_size, num_heads, seq_len, d_v)
    
    # Create an attention module
    attention_layer = ScaledDotProductAttention(dropout=0.1)
    
    # Forward pass
    output, attn_weights = attention_layer(q, k, v)
    
    print("Output shape:", output.shape)
    print("Attention weights shape:", attn_weights.shape)
\`\`\`

This implementation demonstrates the core concepts of attention mechanisms and their practical application in deep learning models.

### Optimization Algorithms

Optimization algorithms are the engines that power model training. They iteratively adjust the model's parameters (weights and biases) to minimize a loss function.

#### Adam: Adaptive Moment Estimation

**Adam** is arguably the most popular optimization algorithm in deep learning today. It combines the ideas of both Momentum (first-order moment) and RMSprop (second-order moment).

The Adam update rules are:
$$m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) \\nabla L(w_t)$$
$$v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) (\\nabla L(w_t))^2$$

With bias correction:
$$\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}$$
$$\\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$$

Final parameter update:
$$w_{t+1} = w_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$$

Default values: $\\beta_1=0.9$, $\\beta_2=0.999$, and $\\epsilon=10^{-8}$.`;

      this.savePost(
        'LLM Base Knowledge - Foundations',
        sampleContent,
        'Engineering Architecture'
      );
    }
  }
} 