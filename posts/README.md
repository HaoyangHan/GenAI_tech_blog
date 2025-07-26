# Blog Posts Directory

This directory contains all blog posts for the GenAI Tech Blog. Posts are written in Markdown with YAML front matter for metadata.

## Quick Start

### Adding a New Post

1. **Create a new `.md` file**:
   ```bash
   touch posts/my-new-post.md
   ```

2. **Add front matter and content**:
   ```markdown
   ---
   title: "My Amazing RAG Implementation"
   category: "Engineering Architecture"
   date: "2024-01-20"
   summary: "Learn how to build a production-ready RAG system"
   slug: "amazing-rag-implementation"
   tags: ["RAG", "Implementation", "Tutorial"]
   author: "Your Name"
   ---

   # My Amazing RAG Implementation

   Your content goes here...
   ```

3. **Start the dev server** to see your post:
   ```bash
   npm run dev
   ```

4. **Visit** `http://localhost:4000` to see your new post!

## Directory Structure

```
posts/
├── README.md                          # This file
├── example-post.md                    # Example blog post
├── getting-started-with-rag.md        # Sample post with full content
└── assets/                           # Images and other assets
    ├── .gitkeep                      # Keeps directory in git
    ├── rag-architecture.jpg          # Referenced in posts
    └── your-images-here.png           # Add your assets here
```

## Front Matter Fields

### Required Fields
- `title`: The display title of your post
- `category`: Must be one of the supported categories
- `date`: Publication date in YYYY-MM-DD format

### Optional Fields
- `summary`: Brief description (auto-generated if not provided)
- `slug`: URL-friendly identifier (auto-generated from title if not provided)
- `tags`: Array of tags for the post
- `author`: Author name (defaults to "Anonymous")

### Supported Categories
- Business Objective
- Engineering Architecture
- Ingestion
- Retrieval
- Generation
- Evaluation
- Prompt Tuning
- Agentic Workflow

## Adding Images

1. **Place images** in the `assets/` directory:
   ```bash
   cp my-diagram.jpg posts/assets/
   ```

2. **Reference in markdown** using relative paths:
   ```markdown
   ![RAG Architecture](./assets/my-diagram.jpg)
   ```

## Tips

- **File naming**: Use kebab-case like `my-blog-post.md`
- **Image optimization**: Keep images under 1MB for better performance
- **Alt text**: Always provide descriptive alt text for accessibility
- **Code blocks**: Use triple backticks with language specification for syntax highlighting

## Examples

Check out `example-post.md` and `getting-started-with-rag.md` for complete examples of properly formatted blog posts.

## Need Help?

See the full documentation at [`Docs/blog-post-management.md`](../Docs/blog-post-management.md) for detailed information about:
- Advanced features
- Troubleshooting
- Asset management
- Programmatic post creation

---

Happy blogging! 🚀 