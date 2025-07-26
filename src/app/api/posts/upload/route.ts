import { NextRequest, NextResponse } from 'next/server';
import { FileBlogService } from '@/lib/file-blog-service';
import fs from 'fs';
import path from 'path';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const title = formData.get('title') as string;
    const content = formData.get('content') as string;
    const category = formData.get('category') as string;

    if (!title || !content || !category) {
      return NextResponse.json(
        { error: 'Missing required fields: title, content, or category' },
        { status: 400 }
      );
    }

    // Generate a unique filename from title
    const slug = title
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .replace(/\s+/g, '-')
      .substring(0, 50);
    
    const fileName = `${slug}.md`;
    const filePath = path.join(process.cwd(), 'posts', fileName);

    // Check if file already exists
    if (fs.existsSync(filePath)) {
      const timestamp = Date.now();
      const uniqueFileName = `${slug}-${timestamp}.md`;
      const uniqueFilePath = path.join(process.cwd(), 'posts', uniqueFileName);
      
      // Create the markdown file with front matter
      const frontMatter = `---
title: "${title}"
category: "${category}"
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
        slug: `${slug}-${timestamp}`
      });
    } else {
      // Create the markdown file with front matter
      const frontMatter = `---
title: "${title}"
category: "${category}"
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
        slug: slug
      });
    }

  } catch (error) {
    console.error('Error uploading post:', error);
    return NextResponse.json(
      { error: 'Failed to upload post' },
      { status: 500 }
    );
  }
} 