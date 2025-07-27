#!/usr/bin/env python3
"""
Script to add frontmatter headers to all markdown files in llm_foundation_posts directory.
"""

import os
import re
from pathlib import Path
from datetime import datetime

def clean_filename_to_title(filename: str) -> str:
    """Convert filename to a clean title format."""
    # Remove .md extension
    name = filename.replace('.md', '')
    
    # Convert patterns like "1_1_foundations" to "1.1 Foundations"
    name = re.sub(r'(\d+)_(\d+)', r'\1.\2', name)
    
    # Replace underscores and hyphens with spaces
    name = re.sub(r'[-_]', ' ', name)
    
    # Title case each word
    words = name.split()
    title_words = []
    for word in words:
        # Keep acronyms in uppercase
        if word.isupper() and len(word) <= 4:
            title_words.append(word)
        else:
            title_words.append(word.capitalize())
    
    return ' '.join(title_words)

def create_slug_from_title(title: str) -> str:
    """Create a URL-friendly slug from title."""
    slug = title.lower()
    slug = re.sub(r'[^a-z0-9\s\-]', '', slug)
    slug = re.sub(r'[\s\-]+', '-', slug)
    slug = slug.strip('-')
    return slug

def generate_summary(title: str) -> str:
    """Generate a summary based on the title."""
    if any(keyword in title.lower() for keyword in ['llm', 'language model', 'transformer', 'attention']):
        return f"Deep dive into {title.lower()} covering theoretical foundations and practical applications in modern large language models and generative AI systems."
    elif any(keyword in title.lower() for keyword in ['probability', 'statistics', 'distribution', 'metrics']):
        return f"Comprehensive guide to {title.lower()} including mathematical foundations, statistical theory, and practical applications in data science."
    elif any(keyword in title.lower() for keyword in ['machine learning', 'ml', 'sklearn', 'pandas']):
        return f"Essential knowledge on {title.lower()} covering core concepts, algorithms, and hands-on implementation techniques for data science practitioners."
    else:
        return f"Data science foundations covering {title.lower()} with theoretical insights and practical applications."

def add_frontmatter_to_file(file_path: Path) -> bool:
    """Add frontmatter to a markdown file if it doesn't already have it."""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if it already has frontmatter
        if content.strip().startswith('---'):
            print(f"Skipping {file_path.name} - already has frontmatter")
            return False
        
        # Generate frontmatter data
        filename = file_path.name
        title = clean_filename_to_title(filename)
        slug = create_slug_from_title(title)
        summary = generate_summary(title)
        
        # Create frontmatter
        frontmatter = f"""---
title: "{title}"
category: "Statistical Deep Dive"
date: "July 27, 2025"
summary: "{summary}"
slug: "{slug}"
tags: ["data science", "Generative AI", "Math Foundations"]
author: "Haoyang Han"
---

"""
        
        # Combine frontmatter with existing content
        new_content = frontmatter + content
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"Added frontmatter to {file_path.name}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to process all markdown files."""
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    llm_posts_dir = project_root / "llm_foundation_posts"
    
    if not llm_posts_dir.exists():
        print(f"Directory {llm_posts_dir} does not exist!")
        return
    
    print(f"Processing markdown files in {llm_posts_dir}")
    
    # Find all markdown files recursively
    md_files = list(llm_posts_dir.rglob("*.md"))
    
    if not md_files:
        print("No markdown files found!")
        return
    
    print(f"Found {len(md_files)} markdown files")
    
    processed_count = 0
    skipped_count = 0
    
    for md_file in md_files:
        if add_frontmatter_to_file(md_file):
            processed_count += 1
        else:
            skipped_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count} files")
    print(f"Skipped: {skipped_count} files")

if __name__ == "__main__":
    main() 