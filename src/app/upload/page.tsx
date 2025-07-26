'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useDropzone } from 'react-dropzone';
import { Upload, File, X, Check } from 'lucide-react';
import Header from '@/components/layout/Header';
import { BlogCategory, BLOG_CATEGORIES, UploadFormData } from '@/types';

export default function UploadPage() {
  const router = useRouter();
  const [formData, setFormData] = useState<UploadFormData>({
    title: '',
    category: 'Engineering Architecture',
    file: null,
  });
  const [uploading, setUploading] = useState(false);
  const [success, setSuccess] = useState(false);

  const parseFilenameTitle = (filename: string): string => {
    const nameWithoutExt = filename.replace('.md', '');
    
    // Convert patterns like "1_1_foundations" to "1.1.foundations"
    const converted = nameWithoutExt.replace(/(\d+)_(\d+)/g, '$1.$2');
    
    // Convert remaining underscores and hyphens to spaces and title case
    return converted
      .replace(/[-_]/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const onDrop = (acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setFormData(prev => ({ ...prev, file }));
      
      // Auto-generate title from filename using enhanced parsing
      const parsedTitle = parseFilenameTitle(file.name);
      setFormData(prev => ({ ...prev, title: parsedTitle }));
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
    
    if (!formData.file) {
      alert('Please select a markdown file to upload.');
      return;
    }

    setUploading(true);

    try {
      // Create FormData for the API request (upload-first approach)
      const uploadFormData = new FormData();
      uploadFormData.append('file', formData.file);
      
      // Optional: include title and category if user wants to override
      if (formData.title) {
        uploadFormData.append('title', formData.title);
      }
      if (formData.category) {
        uploadFormData.append('category', formData.category);
      }

      // Send to the upload API endpoint with file upload support
      const response = await fetch('/api/posts/upload', {
        method: 'POST',
        body: uploadFormData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }

      const result = await response.json();
      console.log('Upload successful:', result);
      
      // Show success message with parsed details
      if (result.uploadType === 'file') {
        alert(`✅ File uploaded successfully!\n\nTitle: ${result.title}\nCategory: ${result.category}\nSlug: ${result.slug}`);
      }
      
      setSuccess(true);
      setTimeout(() => {
        router.push('/');
      }, 2000);
    } catch (error) {
      console.error('Upload failed:', error);
      alert(`Failed to upload the blog post: ${error instanceof Error ? error.message : 'Unknown error'}`);
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
              Post Title <span className="text-gray-500 font-normal">(optional - auto-parsed from filename)</span>
            </label>
            <input
              type="text"
              id="title"
              value={formData.title}
              onChange={(e) => setFormData(prev => ({ ...prev, title: e.target.value }))}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent"
              placeholder="Will be auto-generated from filename (e.g., '1_1_foundations.md' → '1.1 Foundations')"
            />
            {formData.file && (
              <p className="mt-1 text-sm text-gray-600">
                📝 Parsed from "{formData.file.name}": <strong>{formData.title}</strong>
              </p>
            )}
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
            disabled={uploading || !formData.file}
            className="w-full bg-gray-900 text-white py-3 px-6 rounded-lg font-medium hover:bg-gray-800 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {uploading ? 'Uploading...' : 'Upload Markdown File'}
          </button>
        </form>
      </main>
    </div>
  );
} 