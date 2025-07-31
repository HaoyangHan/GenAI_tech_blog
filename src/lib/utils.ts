import { clsx, type ClassValue } from 'clsx';

/**
 * Utility function to merge class names using clsx
 */
export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

/**
 * Convert a title to a URL-friendly slug
 */
export function titleToSlug(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9 ]/g, '')
    .replace(/\s+/g, '-')
    .trim();
}

/**
 * Generate a unique ID based on timestamp and random number
 */
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Format date to readable string
 */
export function formatDate(date: Date): string {
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

/**
 * Extract summary from markdown content (first paragraph)
 */
export function extractSummary(content: string, maxLength: number = 150): string {
  // Remove markdown syntax and get first paragraph
  const text = content
    .replace(/#{1,6}\s+/g, '') // Remove headers
    .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold
    .replace(/\*(.*?)\*/g, '$1') // Remove italic
    .replace(/`(.*?)`/g, '$1') // Remove inline code
    .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1') // Remove links
    .split('\n\n')[0] // Get first paragraph
    .trim();

  if (text.length <= maxLength) {
    return text;
  }

  return text.substring(0, maxLength).trim() + '...';
}

/**
 * Convert a category name to a URL-safe slug
 */
export function categoryToSlug(categoryName: string): string {
  return categoryName.toLowerCase().replace(/\s+/g, '-');
}

/**
 * Convert a URL slug back to a category name with proper acronym handling
 */
export function slugToCategory(slug: string): string {
  return decodeURIComponent(slug)
    .split('-')
    .map(word => {
      // Handle common acronyms
      const upperWord = word.toUpperCase();
      if (['LLM', 'RAG', 'AI', 'ML', 'NLP', 'API', 'UI', 'UX'].includes(upperWord)) {
        return upperWord;
      }
      return word.charAt(0).toUpperCase() + word.slice(1);
    })
    .join(' ');
} 