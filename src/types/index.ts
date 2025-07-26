export interface BlogPost {
  id: string;
  title: string;
  content: string;
  category: BlogCategory;
  date: Date;
  slug: string;
  summary?: string;
  tags?: string[];
  author?: string;
}

export type BlogCategory = 
  | 'All'
  | 'Business Objective'
  | 'Engineering Architecture'
  | 'Ingestion'
  | 'Retrieval'
  | 'Generation'
  | 'Evaluation'
  | 'Prompt Tuning'
  | 'Agentic Workflow'
  | 'GenAI Knowledge'
  | 'Uncategorized';

export const BLOG_CATEGORIES: BlogCategory[] = [
  'All',
  'Business Objective',
  'Engineering Architecture', 
  'Ingestion',
  'Retrieval',
  'Generation',
  'Evaluation',
  'Prompt Tuning',
  'Agentic Workflow',
  'GenAI Knowledge',
  'Uncategorized',
];

export interface UploadFormData {
  title: string;
  category: Exclude<BlogCategory, 'All'>;
  file: File | null;
} 