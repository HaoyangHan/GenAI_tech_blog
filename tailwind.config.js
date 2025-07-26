/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['Menlo', 'Monaco', 'Courier New', 'monospace'],
      },
      typography: {
        DEFAULT: {
          css: {
            maxWidth: '800px',
            color: '#1a1a1a',
            lineHeight: '1.6',
            'code::before': {
              content: '""',
            },
            'code::after': {
              content: '""',
            },
            code: {
              backgroundColor: '#f4f4f4',
              padding: '0.25rem 0.375rem',
              borderRadius: '0.25rem',
              fontWeight: '400',
            },
            pre: {
              backgroundColor: '#f4f4f4',
              padding: '1rem',
              borderRadius: '0.5rem',
              overflow: 'auto',
            },
            'pre code': {
              backgroundColor: 'transparent',
              padding: '0',
            },
          },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}; 