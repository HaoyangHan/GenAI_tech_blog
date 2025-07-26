import { NextRequest, NextResponse } from 'next/server';
import { FileBlogService } from '@/lib/file-blog-service';
import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

// Parse filename to extract a clean title (converts underscores to dots in numeric patterns)
function parseFilenameTitle(filename: string): string {
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

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    
    // Check if this is a file upload (upload-first approach) or form submission
    const file = formData.get('file') as File | null;
    
    if (file) {
      // Upload-first approach: Parse uploaded markdown file
      const filename = file.name;
      const content = await file.text();
      
      if (!filename.endsWith('.md')) {
        return NextResponse.json(
          { error: 'Only markdown files (.md) are supported' },
          { status: 400 }
        );
      }

      // Parse front matter to extract metadata
      const { data, content: markdownContent } = matter(content);
      
      // Extract title from front matter or parse from filename
      const title = data.title || parseFilenameTitle(filename);
      const category = data.category || 'GenAI Knowledge'; // Default category
      const summary = data.summary || 'Uploaded via web interface';
      const author = data.author || 'Haoyang Han';
      const tags = data.tags || [];
      
      // Generate slug from filename or title
      const slug = data.slug || filename.replace('.md', '').toLowerCase().replace(/[^a-z0-9\s]/g, '').replace(/\s+/g, '-');
      
      // Validate category
      const validCategories = ['Business Objective', 'Engineering Architecture', 'Ingestion', 'Retrieval', 'Generation', 'Evaluation', 'Prompt Tuning', 'Agentic Workflow', 'GenAI Knowledge'];
      const finalCategory = validCategories.includes(category) ? category : 'GenAI Knowledge';
      
      // Create category folder if it doesn't exist
      const categoryPath = path.join(process.cwd(), 'posts', finalCategory);
      if (!fs.existsSync(categoryPath)) {
        fs.mkdirSync(categoryPath, { recursive: true });
        // Also create assets folder
        const assetsPath = path.join(categoryPath, 'assets');
        fs.mkdirSync(assetsPath, { recursive: true });
      }
      
      // Create the file path
      const fileName = `${slug}.md`;
      const filePath = path.join(categoryPath, fileName);
      
      // Check if file already exists
      if (fs.existsSync(filePath)) {
        const timestamp = Date.now();
        const uniqueFileName = `${slug}-${timestamp}.md`;
        const uniqueFilePath = path.join(categoryPath, uniqueFileName);
        
        // Create the markdown file with front matter
        const frontMatter = `---
title: "${title}"
category: "${finalCategory}"
date: "${data.date || new Date().toISOString().split('T')[0]}"
summary: "${summary}"
slug: "${slug}-${timestamp}"
tags: ${JSON.stringify(tags)}
author: "${author}"
---

${markdownContent}`;

        fs.writeFileSync(uniqueFilePath, frontMatter);
        
        return NextResponse.json({
          success: true,
          message: 'Markdown file uploaded successfully',
          filename: uniqueFileName,
          slug: `${slug}-${timestamp}`,
          title: title,
          category: finalCategory,
          uploadType: 'file'
        });
      } else {
        // Create the markdown file with front matter
        const frontMatter = `---
title: "${title}"
category: "${finalCategory}"
date: "${data.date || new Date().toISOString().split('T')[0]}"
summary: "${summary}"
slug: "${slug}"
tags: ${JSON.stringify(tags)}
author: "${author}"
---

${markdownContent}`;

        fs.writeFileSync(filePath, frontMatter);
        
        return NextResponse.json({
          success: true,
          message: 'Markdown file uploaded successfully',
          filename: fileName,
          slug: slug,
          title: title,
          category: finalCategory,
          uploadType: 'file'
        });
      }
    } else {
      // Traditional form-based upload
      const title = formData.get('title') as string;
      const content = formData.get('content') as string;
      const category = formData.get('category') as string;

      if (!title || !content || !category) {
        return NextResponse.json(
          { error: 'Missing required fields: title, content, or category' },
          { status: 400 }
        );
      }

      // Validate category
      const validCategories = ['Business Objective', 'Engineering Architecture', 'Ingestion', 'Retrieval', 'Generation', 'Evaluation', 'Prompt Tuning', 'Agentic Workflow', 'GenAI Knowledge'];
      const finalCategory = validCategories.includes(category) ? category : 'GenAI Knowledge';
      
      // Create category folder if it doesn't exist
      const categoryPath = path.join(process.cwd(), 'posts', finalCategory);
      if (!fs.existsSync(categoryPath)) {
        fs.mkdirSync(categoryPath, { recursive: true });
        // Also create assets folder
        const assetsPath = path.join(categoryPath, 'assets');
        fs.mkdirSync(assetsPath, { recursive: true });
      }

      // Generate a unique filename from title
      const slug = title
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, '')
        .replace(/\s+/g, '-')
        .substring(0, 50);
      
      const fileName = `${slug}.md`;
      const filePath = path.join(categoryPath, fileName);

      // Check if file already exists
      if (fs.existsSync(filePath)) {
        const timestamp = Date.now();
        const uniqueFileName = `${slug}-${timestamp}.md`;
        const uniqueFilePath = path.join(categoryPath, uniqueFileName);
        
        // Create the markdown file with front matter
        const frontMatter = `---
title: "${title}"
category: "${finalCategory}"
date: "${new Date().toISOString().split('T')[0]}"
summary: "Uploaded via web interface"
slug: "${slug}-${timestamp}"
author: "Haoyang Han"
---

${content}`;

        fs.writeFileSync(uniqueFilePath, frontMatter);
        
        return NextResponse.json({
          success: true,
          message: 'Post uploaded successfully',
          filename: uniqueFileName,
          slug: `${slug}-${timestamp}`,
          uploadType: 'form'
        });
      } else {
        // Create the markdown file with front matter
        const frontMatter = `---
title: "${title}"
category: "${finalCategory}"
date: "${new Date().toISOString().split('T')[0]}"
summary: "Uploaded via web interface"
slug: "${slug}"
author: "Haoyang Han"
---

${content}`;

        fs.writeFileSync(filePath, frontMatter);
        
        return NextResponse.json({
          success: true,
          message: 'Post uploaded successfully',
          filename: fileName,
          slug: slug,
          uploadType: 'form'
        });
      }
    }

  } catch (error) {
    console.error('Error uploading post:', error);
    return NextResponse.json(
      { error: 'Failed to upload post' },
      { status: 500 }
    );
  }
} 