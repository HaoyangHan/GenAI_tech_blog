'use client';

import { BlogCategory, BLOG_CATEGORIES } from '@/types';
import { cn } from '@/lib/utils';

interface CategoryFilterProps {
  selectedCategory: BlogCategory;
  onCategoryChange: (category: BlogCategory) => void;
  availableCategories?: BlogCategory[];
}

export default function CategoryFilter({ 
  selectedCategory, 
  onCategoryChange, 
  availableCategories = BLOG_CATEGORIES 
}: CategoryFilterProps) {
  return (
    <div className="flex flex-wrap gap-2 mb-8">
      {availableCategories.map((category) => (
        <button
          key={category}
          onClick={() => onCategoryChange(category)}
          className={cn(
            'px-4 py-2 rounded-full text-sm font-medium transition-colors',
            selectedCategory === category
              ? 'bg-gray-900 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          )}
        >
          {category}
        </button>
      ))}
    </div>
  );
} 