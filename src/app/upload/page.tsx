'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useDropzone } from 'react-dropzone';
import { Upload, File, X, Check } from 'lucide-react';
import Header from '@/components/layout/Header';
import { BlogCategory, BLOG_CATEGORIES, UploadFormData } from '@/types';
import { BlogService } from '@/lib/blog-service';

export default function UploadPage() {
  const router = useRouter();
  const [formData, setFormData] = useState<UploadFormData>({
    title: '',
    category: 'Engineering Architecture',
    file: null,
  });
  const [uploading, setUploading] = useState(false);
  const [success, setSuccess] = useState(false);

  const onDrop = (acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setFormData(prev => ({ ...prev, file }));
      
      // Auto-generate title from filename if empty
      if (!formData.title) {
        const fileName = file.name.replace('.md', '').replace(/[-_]/g, ' ');
        setFormData(prev => ({ ...prev, title: fileName }));
      }
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/markdown': ['.md'],
      'text/plain': ['.txt'],
    },
    multiple: false,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.file || !formData.title) {
      alert('Please provide both a title and select a file.');
      return;
    }

    setUploading(true);

    try {
      const content = await formData.file.text();
      await BlogService.savePost(formData.title, content, formData.category);
      
      setSuccess(true);
      setTimeout(() => {
        router.push('/');
      }, 2000);
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Failed to upload the blog post. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const removeFile = () => {
    setFormData(prev => ({ ...prev, file: null }));
  };

  if (success) {
    return (
      <div className="min-h-screen bg-white">
        <Header />
        <main className="max-w-2xl mx-auto px-6 py-12">
          <div className="text-center">
            <div className="mx-auto w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-4">
              <Check className="w-8 h-8 text-green-600" />
            </div>
            <h1 className="text-2xl font-bold text-gray-900 mb-2">Upload Successful!</h1>
            <p className="text-gray-600 mb-4">Your blog post has been uploaded successfully.</p>
            <p className="text-sm text-gray-500">Redirecting to homepage...</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      <main className="max-w-2xl mx-auto px-6 py-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-8">Upload New Post</h1>
        
        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Title Input */}
          <div>
            <label htmlFor="title" className="block text-sm font-medium text-gray-700 mb-2">
              Post Title
            </label>
            <input
              type="text"
              id="title"
              value={formData.title}
              onChange={(e) => setFormData(prev => ({ ...prev, title: e.target.value }))}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent"
              placeholder="Enter the title of your blog post"
              required
            />
          </div>

          {/* Category Selection */}
          <div>
            <label htmlFor="category" className="block text-sm font-medium text-gray-700 mb-2">
              Category
            </label>
            <select
              id="category"
              value={formData.category}
              onChange={(e) => setFormData(prev => ({ ...prev, category: e.target.value as Exclude<BlogCategory, 'All'> }))}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent"
            >
              {BLOG_CATEGORIES.filter(cat => cat !== 'All').map((category) => (
                <option key={category} value={category}>
                  {category}
                </option>
              ))}
            </select>
          </div>

          {/* File Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Markdown File
            </label>
            
            {!formData.file ? (
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive 
                    ? 'border-gray-900 bg-gray-50' 
                    : 'border-gray-300 hover:border-gray-400'
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="mx-auto w-12 h-12 text-gray-400 mb-4" />
                <p className="text-lg font-medium text-gray-700 mb-2">
                  {isDragActive ? 'Drop your markdown file here' : 'Upload your markdown file'}
                </p>
                <p className="text-sm text-gray-500">
                  Drag & drop a .md file here, or click to browse
                </p>
              </div>
            ) : (
              <div className="border border-gray-300 rounded-lg p-4 flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <File className="w-6 h-6 text-gray-600" />
                  <div>
                    <p className="font-medium text-gray-900">{formData.file.name}</p>
                    <p className="text-sm text-gray-500">
                      {(formData.file.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={removeFile}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            )}
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={uploading || !formData.file || !formData.title}
            className="w-full bg-gray-900 text-white py-3 px-6 rounded-lg font-medium hover:bg-gray-800 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {uploading ? 'Uploading...' : 'Upload Post'}
          </button>
        </form>
      </main>
    </div>
  );
} 