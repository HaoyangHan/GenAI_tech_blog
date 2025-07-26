# Getting Started with GenAI Tech Blog

## Quick Start Guide

This document provides step-by-step instructions for setting up and running the GenAI Tech Blog locally.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js 18.17.0 or higher** (recommended: Node.js 20.19.4)
- **npm** (comes with Node.js) or **yarn**
- **Git** for version control

### Node.js Installation

If you need to upgrade Node.js, use Homebrew on macOS:

```bash
# Install Node.js 20 (LTS)
brew install node@20

# Add to your PATH (for current session)
export PATH="/opt/homebrew/opt/node@20/bin:$PATH"

# Add to your shell profile permanently
echo 'export PATH="/opt/homebrew/opt/node@20/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Verify installation:
```bash
node --version  # Should show v20.19.4 or higher
npm --version   # Should show 10.8.2 or higher
```

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd GenAI_tech_blog
```

### 2. Install Dependencies

```bash
npm install
```

This will install all required packages including:
- Next.js 14+ framework
- TypeScript for type safety
- TailwindCSS for styling
- Markdown processing libraries
- File upload utilities

### 3. Environment Setup (Optional)

Copy the environment example file:
```bash
cp .env.example .env.local
```

Update any necessary environment variables in `.env.local`.

## Running the Application

### Development Mode

Start the development server on port 4000:

```bash
npm run dev
```

The application will be available at: **http://localhost:4000**

### Production Build

Build the application for production:

```bash
npm run build
```

Start the production server:

```bash
npm start
```

### Type Checking

Run TypeScript type checking:

```bash
npm run type-check
```

### Linting

Run ESLint to check code quality:

```bash
npm run lint
```

## Project Structure

```
GenAI_tech_blog/
├── src/
│   ├── app/                 # Next.js App Router pages
│   │   ├── globals.css     # Global styles
│   │   ├── layout.tsx      # Root layout
│   │   ├── page.tsx        # Homepage (blog listing)
│   │   ├── post/[slug]/    # Dynamic blog post routes
│   │   └── upload/         # File upload page
│   ├── components/         # Reusable React components
│   │   ├── layout/        # Layout components
│   │   └── ui/            # UI components
│   ├── lib/               # Utility functions and services
│   └── types/             # TypeScript type definitions
├── assets/                # Sample content and static files
├── Docs/                  # Project documentation
└── Configuration files...
```

## Using the Blog

### 1. Viewing Articles

- Open http://localhost:4000
- Browse all articles on the homepage
- Use category filters to find specific topics
- Click article titles to read full content

### 2. Uploading New Posts

1. Click the **"Upload"** button in the header
2. Fill in the post details:
   - **Title**: Descriptive title for your post
   - **Category**: Select from RAG implementation topics
   - **File**: Drag & drop or select a markdown (.md) file
3. Click **"Upload Post"** to save

### 3. Supported Categories

- Business Objective
- Engineering Architecture
- Ingestion
- Retrieval
- Generation
- Evaluation
- Prompt Tuning
- Agentic Workflow

## Troubleshooting

### Common Issues

**1. Node.js Version Error**
```
Error: You are using Node.js X.X.X. For Next.js, Node.js version >= v18.17.0 is required.
```
Solution: Upgrade Node.js using the instructions above.

**2. Port Already in Use**
```
Error: Port 4000 is already in use
```
Solution: Kill the process using the port or use a different port:
```bash
# Kill process on port 4000
lsof -ti:4000 | xargs kill -9

# Or run on different port
npm run dev -- -p 4001
```

**3. Build Errors**
```
Error: Failed to compile
```
Solution: Check for TypeScript/ESLint errors and fix them:
```bash
npm run type-check
npm run lint
```

### Getting Help

- Check the main [README.md](../README.md) for detailed documentation
- Review the [Design Documentation](./design_idea.md)
- Ensure all dependencies are properly installed
- Verify Node.js version compatibility

## Development Tips

1. **Hot Reload**: The development server automatically reloads when you make changes
2. **Type Safety**: Use TypeScript types for better development experience
3. **Code Quality**: Run linting before committing changes
4. **Component Structure**: Follow the atomic design principles
5. **Performance**: Use Next.js built-in optimizations

## Next Steps

After getting the application running:

1. Upload your first markdown blog post
2. Customize the styling in `src/app/globals.css`
3. Add new categories in `src/types/index.ts`
4. Extend functionality by adding new components
5. Deploy to production (see README.md for deployment options)

---

**Happy Blogging!** 🚀 