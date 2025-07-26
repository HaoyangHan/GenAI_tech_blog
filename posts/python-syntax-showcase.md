---
title: "Python Syntax Highlighting Showcase"
category: "GenAI Knowledge"
date: "2024-01-26"
summary: "Demonstrating enhanced Python syntax highlighting with various code examples"
tags: ["Python", "Syntax", "Highlighting", "Code"]
author: "Haoyang Han"
---

# Python Syntax Highlighting Showcase

This post demonstrates the enhanced Python syntax highlighting capabilities of our blog system. Python is now the default language for code blocks, and we've added beautiful syntax highlighting!

## Basic Python Syntax

Here's a simple function without specifying the language (defaults to Python):

```
def greet(name):
    """
    A simple greeting function that demonstrates Python syntax highlighting.
    """
    if name:
        return f"Hello, {name}! Welcome to our blog."
    else:
        return "Hello, anonymous visitor!"

# Example usage
user_name = "Haoyang"
message = greet(user_name)
print(message)
```

## Advanced Python Examples

### Object-Oriented Programming

```python
class RAGSystem:
    """
    A Retrieval-Augmented Generation system for processing documents.
    """
    
    def __init__(self, model_name: str, chunk_size: int = 512):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.documents = []
        self.embeddings = None
    
    def add_document(self, text: str, metadata: dict = None):
        """Add a document to the RAG system."""
        chunks = self._chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            doc_info = {
                'text': chunk,
                'chunk_id': i,
                'metadata': metadata or {}
            }
            self.documents.append(doc_info)
    
    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks of specified size."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    async def query(self, question: str, top_k: int = 5):
        """Query the RAG system with a question."""
        try:
            # Vector similarity search
            similar_docs = self._find_similar_documents(question, top_k)
            
            # Generate response using LLM
            context = '\n'.join([doc['text'] for doc in similar_docs])
            response = await self._generate_response(question, context)
            
            return {
                'answer': response,
                'sources': similar_docs,
                'confidence': 0.95
            }
        except Exception as e:
            raise RuntimeError(f"Query failed: {str(e)}")
```

### Data Processing and Analysis

```python
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta

def analyze_blog_performance(posts_data: List[Dict]) -> Dict:
    """
    Analyze blog post performance metrics.
    
    Args:
        posts_data: List of blog post dictionaries with metrics
        
    Returns:
        Dictionary containing analysis results
    """
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(posts_data)
    
    # Calculate key metrics
    metrics = {
        'total_posts': len(df),
        'avg_views': df['views'].mean(),
        'top_categories': df.groupby('category')['views'].sum().sort_values(ascending=False).head(5),
        'engagement_rate': (df['likes'] + df['comments']) / df['views'],
        'trending_posts': df.nlargest(10, 'views')[['title', 'category', 'views']].to_dict('records')
    }
    
    # Time-based analysis
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    
    monthly_stats = df.groupby('month').agg({
        'views': ['sum', 'mean'],
        'likes': 'sum',
        'comments': 'sum'
    }).round(2)
    
    metrics['monthly_trends'] = monthly_stats.to_dict()
    
    # Performance scoring
    df['performance_score'] = (
        df['views'] * 0.4 + 
        df['likes'] * 0.3 + 
        df['comments'] * 0.3
    )
    
    metrics['top_performers'] = df.nlargest(5, 'performance_score')[
        ['title', 'performance_score']
    ].to_dict('records')
    
    return metrics

# Example usage with sample data
sample_posts = [
    {
        'title': 'Advanced RAG Techniques',
        'category': 'GenAI Knowledge',
        'views': 1500,
        'likes': 89,
        'comments': 23,
        'date': '2024-01-15'
    },
    {
        'title': 'Vector Database Comparison',
        'category': 'Engineering Architecture',
        'views': 2300,
        'likes': 156,
        'comments': 45,
        'date': '2024-01-20'
    }
]

analysis_results = analyze_blog_performance(sample_posts)
print(f"Total posts analyzed: {analysis_results['total_posts']}")
```

### Async Programming and Error Handling

```python
import asyncio
import aiohttp
from contextlib import asynccontextmanager
from typing import AsyncGenerator

class APIClient:
    """Asynchronous API client with proper error handling."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[aiohttp.ClientSession, None]:
        """Context manager for HTTP sessions."""
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        
        try:
            yield self._session
        finally:
            # Session cleanup handled elsewhere
            pass
    
    async def fetch_posts(self, category: str = None) -> List[Dict]:
        """Fetch blog posts from API."""
        url = f"{self.base_url}/api/posts"
        params = {'category': category} if category else {}
        
        async with self.get_session() as session:
            try:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    return [
                        {
                            'id': post['id'],
                            'title': post['title'],
                            'category': post['category'],
                            'author': post.get('author', 'Unknown'),
                            'created_at': post['date']
                        }
                        for post in data
                    ]
            
            except aiohttp.ClientError as e:
                raise ConnectionError(f"Failed to fetch posts: {e}")
            except ValueError as e:
                raise ValueError(f"Invalid JSON response: {e}")
    
    async def close(self):
        """Clean up resources."""
        if self._session:
            await self._session.close()
            self._session = None

# Usage example
async def main():
    client = APIClient('http://localhost:4000')
    
    try:
        # Fetch all posts
        all_posts = await client.fetch_posts()
        print(f"Found {len(all_posts)} posts")
        
        # Fetch GenAI Knowledge posts
        genai_posts = await client.fetch_posts('GenAI Knowledge')
        print(f"Found {len(genai_posts)} GenAI posts")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
```

## Other Languages Support

The system also supports other programming languages with syntax highlighting:

### JavaScript/TypeScript

```javascript
// Fetch blog posts with error handling
async function fetchBlogPosts(category = null) {
    try {
        const url = new URL('/api/posts', window.location.origin);
        if (category) {
            url.searchParams.set('category', category);
        }
        
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const posts = await response.json();
        return posts.map(post => ({
            ...post,
            date: new Date(post.date)
        }));
    } catch (error) {
        console.error('Failed to fetch posts:', error);
        return [];
    }
}
```

### SQL

```sql
-- Blog analytics query
SELECT 
    category,
    COUNT(*) as post_count,
    AVG(view_count) as avg_views,
    MAX(created_at) as latest_post
FROM blog_posts 
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
    AND status = 'published'
GROUP BY category
ORDER BY avg_views DESC;
```

### Bash

```bash
#!/bin/bash
# Deploy blog to production

set -e

echo "🚀 Deploying blog to production..."

# Build the application
npm run build

# Run tests
npm test

# Deploy to server
rsync -avz --delete ./build/ user@server:/var/www/blog/

echo "✅ Deployment completed successfully!"
```

## Conclusion

Our enhanced Python syntax highlighting provides:

- **Beautiful colors** for keywords, strings, comments, and functions
- **Python as default** - no need to specify language for Python code
- **Fallback support** - other languages still work great
- **Professional styling** - dark theme with excellent readability

Try uploading your own Python code examples to see the highlighting in action! 