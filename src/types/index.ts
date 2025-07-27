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
  knowledgeBase?: KnowledgeBase; // New field to distinguish between knowledge bases
}

export type KnowledgeBase = 'rag' | 'foundations';

export type BlogCategory = 
  // RAG Implementation Categories
  | 'All'
  | 'Business Objective'
  | 'Engineering Architecture'
  | 'Ingestion'
  | 'Retrieval'
  | 'Generation'
  | 'Evaluation'
  | 'Prompt Tuning'
  | 'Agentic Workflow'
  // Foundation Categories
  | 'LLM Base Knowledge'
  | 'LLM Model Architecture'
  | 'Training Data'
  | 'Fine Tuning'
  | 'RAG'
  | 'LLM Evaluation'
  | 'Traditional ML'
  | 'Statistical Deep Dive' // Keep for backward compatibility
  | 'Uncategorized';

export const RAG_CATEGORIES: BlogCategory[] = [
  'All',
  'Business Objective',
  'Engineering Architecture', 
  'Ingestion',
  'Retrieval',
  'Generation',
  'Evaluation',
  'Prompt Tuning',
  'Agentic Workflow',
];

export const FOUNDATION_CATEGORIES: BlogCategory[] = [
  'All',
  'LLM Base Knowledge',
  'LLM Model Architecture',
  'Training Data',
  'Fine Tuning',
  'RAG',
  'LLM Evaluation',
  'Traditional ML',
];

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
  'LLM Base Knowledge',
  'LLM Model Architecture',
  'Training Data',
  'Fine Tuning',
  'RAG',
  'LLM Evaluation',
  'Traditional ML',
  'Statistical Deep Dive',
  'Uncategorized',
];

export interface UploadFormData {
  title: string;
  category: Exclude<BlogCategory, 'All'>;
  file: File | null;
}

export interface KnowledgeBaseInfo {
  id: KnowledgeBase;
  title: string;
  description: string;
  categories: BlogCategory[];
  path: string;
} 