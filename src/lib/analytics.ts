// Google Analytics utility functions

export const GA_TRACKING_ID = 'G-C2XHG5LGKX';

// Track page views
export const pageview = (url: string) => {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('config', GA_TRACKING_ID, {
      page_location: url,
    });
  }
};

// Track custom events
export const event = (action: string, parameters?: {
  event_category?: string;
  event_label?: string;
  value?: number;
  [key: string]: any;
}) => {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('event', action, parameters);
  }
};

// Track specific blog events
export const trackBlogEvent = {
  viewPost: (postSlug: string, category: string) => {
    event('view_blog_post', {
      event_category: 'Blog',
      event_label: postSlug,
      custom_parameter_category: category,
    });
  },
  
  copyCode: (language: string) => {
    event('copy_code_block', {
      event_category: 'Engagement',
      event_label: language,
    });
  },
  
  copyMath: () => {
    event('copy_math_equation', {
      event_category: 'Engagement',
      event_label: 'LaTeX',
    });
  },
  
  downloadPDF: (fileName: string) => {
    event('download_file', {
      event_category: 'Downloads',
      event_label: fileName,
    });
  },
}; 