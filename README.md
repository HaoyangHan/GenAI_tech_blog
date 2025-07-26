# GenAI Tech Blog - RAG Implementation Details

A minimal, beautifully designed blog website for technical content about RAG (Retrieval-Augmented Generation) implementation. Built with Next.js 14+, TypeScript, and TailwindCSS following modern frontend best practices.

![Blog Preview](./assets/blog-preview.png)

## 🌟 Features

- **Modern Design**: Clean, minimalist interface inspired by OpenAI's research page
- **Markdown Support**: Upload and render markdown files with beautiful typography
- **Category Filtering**: Organize posts by RAG implementation topics
- **Responsive Design**: Mobile-first approach with fluid layouts
- **File Upload**: Drag & drop interface for markdown file uploads
- **Local Storage**: Client-side storage for blog posts (perfect for demos)
- **TypeScript**: Full type safety throughout the application
- **Performance Optimized**: Next.js 14+ with App Router for optimal performance

## 📚 Categories

The blog supports the following RAG implementation categories:

- **Business Objective** - Strategic goals and requirements
- **Engineering Architecture** - System design and infrastructure
- **Ingestion** - Data processing and preparation
- **Retrieval** - Vector search and document retrieval
- **Generation** - LLM integration and response generation
- **Evaluation** - Metrics and performance assessment
- **Prompt Tuning** - Optimization and fine-tuning
- **Agentic Workflow** - AI agent implementations

## 🚀 Getting Started

### Prerequisites

- Node.js 18.17.0 or higher (required for Next.js 14+)
- npm or yarn package manager

> **Note**: This project has been tested with Node.js 20.19.4. If you need to upgrade Node.js, you can install it using Homebrew: `brew install node@20` and add it to your PATH.

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GenAI_tech_blog
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

4. **Open your browser**
   Navigate to [http://localhost:4000](http://localhost:4000)

## 📝 Usage

### Uploading Blog Posts

1. **Navigate to Upload Page**
   - Click the "Upload" button in the header
   - Or visit `/upload` directly

2. **Fill in Post Details**
   - **Title**: Enter a descriptive title for your post
   - **Category**: Select the appropriate RAG implementation category
   - **File**: Drag & drop or select a markdown (.md) file

3. **Submit**
   - Click "Upload Post" to save your blog post
   - You'll be redirected to the homepage to see your new post

### Managing Posts

- **View All Posts**: Homepage displays all posts with category filters
- **Read Individual Posts**: Click on any post title to view the full content
- **Filter by Category**: Use the category buttons to filter posts
- **Navigate**: Use the header navigation to move between pages

## 📁 Project Structure

```
src/
├── app/                     # Next.js App Router pages
│   ├── globals.css         # Global styles and TailwindCSS imports
│   ├── layout.tsx          # Root layout component
│   ├── page.tsx            # Homepage (blog listing)
│   ├── post/[slug]/        # Dynamic blog post pages
│   │   └── page.tsx        # Individual post component
│   └── upload/             # Upload functionality
│       └── page.tsx        # Upload page component
├── components/             # Reusable React components
│   ├── layout/            # Layout components
│   │   └── Header.tsx     # Site header with navigation
│   └── ui/                # UI components
│       ├── BlogPostCard.tsx    # Blog post preview card
│       └── CategoryFilter.tsx  # Category filter buttons
├── lib/                   # Utility functions and services
│   ├── blog-service.ts    # Blog post CRUD operations
│   └── utils.ts           # Helper functions
└── types/                 # TypeScript type definitions
    └── index.ts           # Blog post and category types
```

## 🎨 Design Philosophy

The design follows these principles:

- **Content-First**: Typography and readability are prioritized
- **Minimalist**: Clean interface with generous white space
- **Accessible**: WCAG 2.1 Level AA compliance
- **Responsive**: Mobile-first design with fluid layouts
- **Performance**: Optimized loading and rendering

## 🛠️ Technical Stack

- **Framework**: Next.js 14+ with App Router
- **Language**: TypeScript 5+
- **Styling**: TailwindCSS with Typography plugin
- **Markdown**: Marked.js for parsing and rendering
- **File Upload**: React Dropzone for drag & drop functionality
- **Icons**: Lucide React for consistent iconography
- **Storage**: Browser localStorage (easily replaceable with database)

## 📖 Sample Content

The application comes with a sample blog post demonstrating:
- Mathematical formulas and equations
- Code syntax highlighting
- Proper typography and formatting
- Technical content structure

You can find the sample markdown file at `assets/1_1_foundations.md`.

## 🔧 Configuration

### Adding New Categories

To add new blog categories, update the `BLOG_CATEGORIES` array in `src/types/index.ts`:

```typescript
export const BLOG_CATEGORIES: BlogCategory[] = [
  'All',
  'Business Objective',
  'Engineering Architecture',
  // Add new categories here
  'Your New Category',
];
```

### Customizing Styles

- Global styles: `src/app/globals.css`
- Component styles: Use TailwindCSS classes in components
- Typography: Customize the `.prose-custom` class in globals.css

### Storage Backend

The current implementation uses localStorage. To integrate with a backend:

1. Update `src/lib/blog-service.ts`
2. Replace localStorage methods with API calls
3. Add proper error handling and loading states

## 🚀 Deployment

### Static Deployment (Recommended)

Since this is a client-side application, it can be deployed to any static hosting service:

1. **Build the application**
   ```bash
   npm run build
   npm run export  # If using static export
   ```

2. **Deploy to platforms like:**
   - Vercel (recommended for Next.js)
   - Netlify
   - GitHub Pages
   - AWS S3 + CloudFront

### Server Deployment

For full server-side rendering:

1. **Build the application**
   ```bash
   npm run build
   ```

2. **Start the production server**
   ```bash
   npm start
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Design inspired by OpenAI's research publications page
- Built following modern React and Next.js best practices
- Typography and accessibility guidelines from WCAG 2.1
- Sample content demonstrates LLM and attention mechanism concepts

---

**Note**: This application uses browser localStorage for data persistence. For production use with multiple users, consider implementing a proper backend database and authentication system.

