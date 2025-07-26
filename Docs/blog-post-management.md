# Blog Post Management Guide

This guide explains how to manage blog posts in the GenAI Tech Blog, including file-based storage, assets management, and direct codebase editing.

## Storage Systems

### Current Options

1. **File-Based Storage** (Recommended)
   - Posts stored as `.md` files in the `posts/` directory
   - Assets stored in `posts/assets/` 
   - Metadata managed via front matter
   - Full version control integration

2. **Browser localStorage** (Legacy)
   - Posts stored in browser's localStorage
   - No asset support
   - Data lost when browser storage is cleared

## File-Based Blog Posts

### Directory Structure

```
posts/
├── getting-started-with-rag.md          # Blog post
├── attention-mechanisms-deep-dive.md    # Another post
├── evaluation-metrics-for-rag.md        # Yet another post
└── assets/                              # Assets directory
    ├── rag-architecture.jpg             # Image for posts
    ├── attention-diagram.png             # Diagram
    └── evaluation-results.svg            # Chart/graph
```

### Creating a New Blog Post

#### Method 1: Direct File Creation

1. **Create a new markdown file** in the `posts/` directory:

```bash
touch posts/your-new-post.md
```

2. **Add front matter and content**:

```markdown
---
title: "Your Amazing Blog Post Title"
category: "Engineering Architecture"
date: "2024-01-20"
summary: "A brief summary of what this post covers."
slug: "your-amazing-blog-post"
tags: ["RAG", "LLM", "Implementation"]
author: "Your Name"
---

# Your Amazing Blog Post Title

Your content goes here...

## Section 1

Content with code examples:

```python
def hello_world():
    print("Hello, RAG World!")
```

![Architecture Diagram](./assets/your-diagram.jpg)

## Conclusion

Wrap up your thoughts here.
```

#### Method 2: Using the FileBlogService

```typescript
import { FileBlogService } from '@/lib/file-blog-service';

// Create a new post programmatically
const slug = await FileBlogService.createPost(
  "Advanced RAG Techniques",
  "# Advanced RAG Techniques\n\nYour content here...",
  "Retrieval",
  {
    slug: "advanced-rag-techniques",
    summary: "Exploring cutting-edge RAG implementation strategies",
    tags: ["RAG", "Advanced", "Techniques"],
    author: "RAG Expert"
  }
);
```

### Front Matter Reference

All blog posts should include front matter at the top:

```yaml
---
title: "Post Title"                    # Required: Display title
category: "Engineering Architecture"   # Required: One of the supported categories
date: "2024-01-20"                    # Required: Publication date (YYYY-MM-DD)
summary: "Brief description"          # Optional: Auto-generated if not provided
slug: "url-friendly-slug"             # Optional: Auto-generated from title if not provided
tags: ["tag1", "tag2"]                # Optional: Array of tags
author: "Author Name"                 # Optional: Defaults to "Anonymous"
---
```

#### Supported Categories

- `Business Objective`
- `Engineering Architecture`
- `Ingestion`
- `Retrieval`
- `Generation`
- `Evaluation`
- `Prompt Tuning`
- `Agentic Workflow`

### Working with Assets

#### Adding Images

1. **Place images in the assets directory**:
```bash
cp your-image.jpg posts/assets/
```

2. **Reference in markdown**:
```markdown
![Alt text](./assets/your-image.jpg)
```

#### Supported Asset Types

- **Images**: JPG, PNG, SVG, WebP
- **Documents**: PDF (for download links)
- **Data**: JSON, CSV (for examples)

#### Asset Guidelines

- Use descriptive filenames: `rag-architecture-v2.jpg` not `image1.jpg`
- Optimize images for web (< 1MB recommended)
- Use relative paths: `./assets/filename.jpg`
- Provide meaningful alt text for accessibility

### Best Practices

#### File Naming

- Use kebab-case: `advanced-rag-techniques.md`
- Be descriptive: `evaluation-metrics-comprehensive-guide.md`
- Avoid special characters and spaces

#### Content Organization

```markdown
---
# Front matter here
---

# Main Title (matches front matter title)

Brief introduction paragraph.

## Table of Contents (optional for long posts)

## Section 1: Core Concepts

### Subsection 1.1

Content here...

## Section 2: Implementation

### Code Examples

```python
# Well-commented code
def example_function():
    """
    Clear docstring explaining the function.
    """
    pass
```

### Diagrams and Visuals

![Descriptive Alt Text](./assets/diagram.jpg)

*Caption explaining the diagram*

## Conclusion

## References (if applicable)

1. Paper citations
2. Documentation links
3. External resources
```

## Integration with the Blog System

### Automatic Discovery

The file-based system automatically:

1. **Scans the `posts/` directory** for `.md` files
2. **Parses front matter** for metadata
3. **Generates summaries** if not provided
4. **Sorts posts** by date (newest first)
5. **Creates URL slugs** from filenames if not specified

### Development Workflow

1. **Create/edit** markdown files in `posts/`
2. **Add assets** to `posts/assets/`
3. **Test locally** by running `npm run dev`
4. **Commit changes** to version control
5. **Deploy** - posts are automatically available

### Version Control Benefits

- **Track changes** to blog posts over time
- **Collaborate** on content with pull requests
- **Backup** posts in your repository
- **Branch** for draft posts
- **Review** content changes before publishing

## Migration from localStorage

### Moving Existing Posts

If you have posts in localStorage that you want to convert to files:

1. **Export from browser** (manual copy)
2. **Create corresponding `.md` files** in `posts/`
3. **Add proper front matter**
4. **Test that posts load correctly**

### Hybrid Approach

The system supports both storage methods:
- **Server-side rendering**: Uses file-based posts
- **Client-side**: Falls back to localStorage if needed

## Troubleshooting

### Common Issues

**Posts not appearing:**
- Check file extension is `.md`
- Verify front matter is valid YAML
- Ensure `posts/` directory exists
- Check for TypeScript/build errors

**Images not loading:**
- Verify image is in `posts/assets/`
- Check relative path: `./assets/image.jpg`
- Ensure image file exists and has correct permissions
- Check image format is supported

**Front matter errors:**
- Validate YAML syntax
- Ensure required fields are present
- Check category spelling matches exactly
- Verify date format is YYYY-MM-DD

### Debug Commands

```bash
# List all posts
ls posts/*.md

# Check front matter syntax
head -10 posts/your-post.md

# Verify assets
ls posts/assets/

# Test build
npm run build
```

## Advanced Usage

### Custom Categories

To add new categories, update `src/types/index.ts`:

```typescript
export const BLOG_CATEGORIES: BlogCategory[] = [
  'All',
  'Business Objective',
  'Engineering Architecture',
  // Add your new category here
  'Custom Category',
];
```

### Programmatic Post Creation

Use the FileBlogService for batch operations:

```typescript
// scripts/create-posts.ts
import { FileBlogService } from '../src/lib/file-blog-service';

const posts = [
  { title: "Post 1", content: "...", category: "Ingestion" },
  { title: "Post 2", content: "...", category: "Retrieval" },
];

for (const post of posts) {
  await FileBlogService.createPost(post.title, post.content, post.category);
}
```

### Automation Scripts

Create scripts for common tasks:

```bash
# scripts/new-post.sh
#!/bin/bash
read -p "Post title: " title
slug=$(echo "$title" | tr '[:upper:]' '[:lower:]' | sed 's/ /-/g')
cat > "posts/${slug}.md" << EOF
---
title: "$title"
category: "Engineering Architecture"
date: "$(date +%Y-%m-%d)"
summary: ""
slug: "$slug"
tags: []
author: "$(git config user.name)"
---

# $title

Your content here...
EOF
echo "Created posts/${slug}.md"
```

---

This file-based approach gives you full control over your blog content while maintaining the beautiful frontend interface. You can now manage posts like code - with version control, collaboration, and proper asset management! 