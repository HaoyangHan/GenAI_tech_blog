# How to Add Blog Posts Directly in Codebase

This guide explains how to add blog posts directly by creating markdown files in the codebase, including handling of assets, front matter configuration, and automatic categorization.

## Overview

The GenAI Tech Blog supports two methods for adding content:
1. **Direct Codebase Editing** (Recommended) - Create `.md` files directly in the `posts/` directory
2. **Frontend Upload** (Legacy) - Upload via web interface (stored in localStorage)

This document focuses on the **direct codebase approach**, which provides:
- ✅ **Version Control**: Track changes with Git
- ✅ **Asset Support**: Include images, diagrams, and files
- ✅ **Collaboration**: Team editing with pull requests
- ✅ **Backup**: Content stored in repository
- ✅ **Advanced Formatting**: Full markdown support
- ✅ **Automatic Processing**: Smart categorization and metadata handling

## Quick Start

### 1. Create a New Blog Post

```bash
# Navigate to the posts directory
cd posts/

# Create your new post file
touch my-awesome-rag-post.md
```

### 2. Add Content with Front Matter

Open the file and add front matter + content:

```markdown
---
title: "Advanced RAG Implementation Patterns"
category: "Engineering Architecture"
date: "2024-01-26"
summary: "Exploring advanced patterns for production RAG systems"
slug: "advanced-rag-patterns"
tags: ["RAG", "Architecture", "Production"]
author: "RAG Expert"
---

# Advanced RAG Implementation Patterns

Your blog content goes here...

## Code Examples

```python
def advanced_rag_pipeline():
    # Your implementation
    pass
```

## Diagrams

![Architecture Overview](./assets/rag-architecture.jpg)

## Conclusion

Summary of key insights...
```

### 3. Test Your Post

```bash
# Start the development server
npm run dev

# Visit http://localhost:4000 to see your post
```

## Front Matter Configuration

### Required Fields

```yaml
---
title: "Your Post Title"              # Display title (required)
category: "Engineering Architecture"  # Must be a valid category (required)
date: "2024-01-26"                   # Publication date YYYY-MM-DD (required)
---
```

### Optional Fields

```yaml
---
title: "Your Post Title"
category: "Engineering Architecture"
date: "2024-01-26"
summary: "Brief description of the post"           # Auto-generated if not provided
slug: "your-post-url-slug"                        # Auto-generated from title if not provided
tags: ["RAG", "LLM", "Implementation"]             # Array of tags for categorization
author: "Your Name"                                # Defaults to "Haoyang Han" if not provided
---
```

### Valid Categories

The system supports these predefined categories:

- `Business Objective` - Strategic goals and business requirements
- `Engineering Architecture` - System design and technical architecture
- `Ingestion` - Data processing and document ingestion
- `Retrieval` - Vector search and document retrieval
- `Generation` - LLM integration and response generation
- `Evaluation` - Metrics, testing, and performance assessment
- `Prompt Tuning` - Prompt engineering and optimization
- `Agentic Workflow` - AI agent implementations and workflows
- `GenAI Knowledge` - General AI concepts, fundamentals, and educational content
- `Uncategorized` - **Automatic fallback** for posts without valid categories

## Automatic Categorization

### Posts Without Front Matter

If a markdown file doesn't have front matter, the system automatically:

1. **Extracts title** from the first `# heading` in the content
2. **Assigns category** as "Uncategorized"
3. **Sets date** to current date
4. **Generates slug** from filename
5. **Creates summary** from first paragraph

**Example**: A file named `my-research-notes.md` without front matter:

```markdown
# My Research Notes

This is my research on RAG implementations...

## Key Findings

Important discoveries...
```

Will be automatically processed as:
- **Title**: "My Research Notes"
- **Category**: "Uncategorized"
- **Slug**: "my-research-notes"
- **Date**: Current date
- **Summary**: Auto-extracted from content

### Invalid Categories

If a post has front matter with an invalid category:

```yaml
---
title: "My Post"
category: "Invalid Category Name"  # Not in valid categories list
date: "2024-01-26"
---
```

The system will automatically assign it to **"Uncategorized"** instead of failing.

## Working with Assets

### Directory Structure

```
posts/
├── your-post.md                     # Your blog post
├── another-post.md                  # Another post
└── assets/                         # Assets directory
    ├── rag-architecture.jpg        # Images
    ├── evaluation-chart.png         # Charts and diagrams
    ├── code-example.py              # Code files
    └── research-paper.pdf           # Documents
```

### Adding Images

1. **Place image in assets directory**:
```bash
cp ~/Downloads/my-diagram.jpg posts/assets/
```

2. **Reference in markdown**:
```markdown
![RAG Architecture Overview](./assets/my-diagram.jpg)

*Figure 1: Complete RAG system architecture showing all components*
```

### Supported Asset Types

- **Images**: `.jpg`, `.png`, `.svg`, `.webp`, `.gif`
- **Documents**: `.pdf`, `.docx` (for download links)
- **Data**: `.json`, `.csv`, `.yaml` (for examples)
- **Code**: `.py`, `.js`, `.ts`, `.sql` (for reference)
- **Any file type** your posts need

### Asset Best Practices

```markdown
<!-- Good: Descriptive filename and alt text -->
![RAG Performance Comparison Chart](./assets/rag-performance-comparison-2024.jpg)

<!-- Good: Caption for context -->
![Vector Database Benchmark Results](./assets/vector-db-benchmark.png)
*Benchmark results comparing Pinecone, Weaviate, and Qdrant performance*

<!-- Good: Download link for documents -->
[Download Research Paper](./assets/rag-evaluation-methodology.pdf)

<!-- Bad: Generic names and missing alt text -->
![](./assets/image1.jpg)
```

## File Organization

### Naming Conventions

**Good filenames:**
- `advanced-rag-implementation.md`
- `vector-database-comparison-2024.md`
- `prompt-engineering-best-practices.md`
- `evaluation-metrics-comprehensive-guide.md`

**Avoid:**
- `post1.md` (not descriptive)
- `RAG Implementation.md` (spaces and capitals)
- `my-post!!.md` (special characters)

### Content Structure Template

```markdown
---
title: "Your Descriptive Title"
category: "Appropriate Category"
date: "YYYY-MM-DD"
summary: "Brief, engaging summary of what readers will learn"
slug: "url-friendly-slug"
tags: ["Primary Tag", "Secondary Tag", "Tertiary Tag"]
author: "Your Name"
---

# Your Descriptive Title

Brief introduction paragraph that hooks the reader and explains what they'll learn.

## Table of Contents (for longer posts)

- [Section 1](#section-1)
- [Section 2](#section-2)
- [Conclusion](#conclusion)

## Section 1: Core Concepts

### Subsection 1.1

Content with clear explanations...

### Code Examples

```python
# Well-commented, production-ready code
class RAGSystem:
    def __init__(self, model_name: str):
        """
        Initialize RAG system with specified model.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model = SentenceTransformer(model_name)
    
    def process_documents(self, documents: List[str]) -> np.ndarray:
        """Process documents and return embeddings."""
        return self.model.encode(documents)
```

## Section 2: Implementation Details

### Visual Aids

![System Architecture](./assets/rag-system-architecture.jpg)
*Complete RAG system showing ingestion, storage, retrieval, and generation components*

### Performance Metrics

| Model | Latency (ms) | Accuracy | Memory (GB) |
|-------|-------------|----------|-------------|
| Model A | 150 | 0.85 | 2.1 |
| Model B | 200 | 0.90 | 3.2 |

## Conclusion

Key takeaways:
1. Main insight #1
2. Main insight #2
3. Main insight #3

## Next Steps

- Link to related posts
- Suggested further reading
- Implementation challenges to explore

## References

1. [Paper Title](https://example.com/paper)
2. [Documentation](https://example.com/docs)
3. [GitHub Repository](https://github.com/example/repo)

---

*Published on [date] | Tags: [tag1], [tag2], [tag3]*
```

## Development Workflow

### 1. Local Development

```bash
# Create new post
touch posts/my-new-post.md

# Edit with your favorite editor
code posts/my-new-post.md

# Add any assets
cp ~/Downloads/diagram.jpg posts/assets/

# Test locally
npm run dev
# Visit http://localhost:4000

# Verify your post appears and renders correctly
```

### 2. Version Control Integration

```bash
# Add your changes
git add posts/my-new-post.md posts/assets/diagram.jpg

# Commit with descriptive message
git commit -m "feat: add comprehensive guide to RAG evaluation metrics

- Cover precision, recall, and relevance metrics
- Include benchmark results and comparisons
- Add interactive evaluation framework
- Provide code examples for implementation"

# Push to repository
git push origin main
```

### 3. Collaboration Workflow

```bash
# Create feature branch for your post
git checkout -b post/advanced-rag-techniques

# Work on your post
echo "---
title: \"Advanced RAG Techniques\"
category: \"Retrieval\"
date: \"$(date +%Y-%m-%d)\"
---

# Advanced RAG Techniques

Content here..." > posts/advanced-rag-techniques.md

# Commit and push
git add .
git commit -m "draft: advanced RAG techniques post"
git push origin post/advanced-rag-techniques

# Create pull request for review
# Team members can review content, suggest changes
# Merge when ready
```

## Advanced Features

### 1. Dynamic Slug Generation

If you don't specify a slug, it's auto-generated:

```markdown
---
title: "Advanced RAG Implementation Patterns for Production Systems"
# No slug specified
---
```

Results in slug: `advanced-rag-implementation-patterns-for-production-systems`

### 2. Automatic Summary Generation

If you don't provide a summary, it's extracted from content:

```markdown
---
title: "RAG Evaluation"
# No summary specified
---

# RAG Evaluation

This comprehensive guide covers all aspects of evaluating RAG systems, including precision metrics, recall measurements, and end-to-end performance assessment. We'll explore both automated and human evaluation approaches.

More content follows...
```

Auto-generated summary: "This comprehensive guide covers all aspects of evaluating RAG systems, including precision metrics, recall measurements, and end-to-end performance assessment..."

### 3. Flexible Date Formats

The system accepts various date formats:

```yaml
date: "2024-01-26"           # YYYY-MM-DD (recommended)
date: "2024-01-26T10:30:00"  # ISO format with time
date: "January 26, 2024"     # Natural language (parsed automatically)
```

### 4. Tag-Based Organization

Use tags for cross-cutting concerns:

```yaml
tags: ["RAG", "Production", "Performance", "Tutorial"]
```

Tags help readers find related content across categories.

## Troubleshooting

### Common Issues

**1. Post not appearing on website**
```bash
# Check file extension
ls posts/*.md

# Verify file has content
cat posts/your-post.md

# Check server logs
npm run dev
# Look for any error messages
```

**2. Images not loading**
```bash
# Verify image exists
ls posts/assets/your-image.jpg

# Check relative path in markdown
grep "assets/" posts/your-post.md

# Ensure correct format
# Correct: ![Alt text](./assets/image.jpg)
# Wrong: ![Alt text](/assets/image.jpg)
```

**3. Front matter parsing errors**
```yaml
# Ensure YAML is valid
---
title: "My Post"          # Quotes around strings with special chars
category: Engineering     # No quotes needed for simple strings
date: 2024-01-26         # No quotes needed for dates
tags:                    # Array format
  - RAG
  - LLM
# Alternative array format: tags: ["RAG", "LLM"]
---
```

**4. Post showing as "Uncategorized"**
```yaml
# Check category spelling exactly matches supported categories
category: "Engineering Architecture"  # Correct
category: "engineering architecture"  # Wrong (case sensitive)
category: "Engineering"               # Wrong (incomplete)
```

### Debug Commands

```bash
# List all posts and check naming
ls -la posts/*.md

# Check front matter of specific post
head -20 posts/your-post.md

# Verify assets
ls -la posts/assets/

# Test API directly
curl http://localhost:4000/api/posts | jq '.[0]'

# Check specific post by slug
curl http://localhost:4000/api/posts/your-post-slug | jq '.'
```

## Best Practices Summary

### Content Quality
- ✅ Write clear, descriptive titles
- ✅ Provide helpful summaries
- ✅ Use proper markdown formatting
- ✅ Include relevant code examples
- ✅ Add meaningful images with alt text
- ✅ Structure content with clear headings

### Technical Implementation
- ✅ Use kebab-case for filenames
- ✅ Always include required front matter
- ✅ Place assets in the `assets/` directory
- ✅ Use relative paths for asset references
- ✅ Test locally before committing
- ✅ Write descriptive commit messages

### Collaboration
- ✅ Use feature branches for new posts
- ✅ Request reviews for technical accuracy
- ✅ Keep related assets with the post
- ✅ Update documentation when adding new patterns
- ✅ Tag posts appropriately for discoverability

---

This file-based approach gives you complete control over your blog content while maintaining a beautiful, professional frontend interface. You can now manage blog posts like code - with proper version control, team collaboration, and professional development workflows! 🚀 