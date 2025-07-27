import { useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import CopyButton from '@/components/ui/CopyButton';

export function useEnhancedContent(htmlContent: string) {
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!contentRef.current || !htmlContent) return;

    const container = contentRef.current;

    // Function to extract text from code blocks
    const getCodeText = (element: HTMLElement): string => {
      // If it's a pre > code structure, get the code element's text
      const codeElement = element.querySelector('code');
      if (codeElement) {
        return codeElement.textContent || '';
      }
      return element.textContent || '';
    };

    // Function to extract LaTeX from math elements
    const getMathText = (element: HTMLElement): string => {
      // Look for the original LaTeX in data attributes or reconstruct from KaTeX
      const katexElement = element.querySelector('.katex');
      if (katexElement) {
        // Try to get the original LaTeX from a data attribute if we stored it
        const originalLatex = katexElement.getAttribute('data-latex');
        if (originalLatex) {
          return originalLatex;
        }
        
        // Fallback: extract from the annotation if available
        const annotation = katexElement.querySelector('annotation[encoding="application/x-tex"]');
        if (annotation) {
          return annotation.textContent || '';
        }
      }
      
      // Last resort: return the text content
      return element.textContent || '';
    };

    // Add copy buttons to code blocks
    const codeBlocks = container.querySelectorAll('pre');
    codeBlocks.forEach((block) => {
      if (block.querySelector('.copy-button-container')) return; // Already enhanced
      
      const code = getCodeText(block);
      if (!code.trim()) return;

      // Create container for the copy button
      const buttonContainer = document.createElement('div');
      buttonContainer.className = 'copy-button-container absolute top-2 right-2 z-10';
      
      // Make the pre element relative positioned and add group class
      block.style.position = 'relative';
      block.classList.add('group');
      
      // Create React root and render the copy button
      const root = createRoot(buttonContainer);
      root.render(<CopyButton text={code} size={14} />);
      
      block.appendChild(buttonContainer);
    });

    // Add copy buttons to inline code
    const inlineCodes = container.querySelectorAll('code:not(pre code)');
    inlineCodes.forEach((code) => {
      if (code.querySelector('.copy-button-container')) return; // Already enhanced
      
      const text = code.textContent || '';
      if (!text.trim() || text.length < 5) return; // Skip very short inline code
      
      // Create container for the copy button
      const buttonContainer = document.createElement('span');
      buttonContainer.className = 'copy-button-container inline-block ml-1 align-middle';
      
      // Add group class to parent for hover effect
      code.classList.add('group', 'relative');
      
      // Create React root and render the copy button
      const root = createRoot(buttonContainer);
      root.render(<CopyButton text={text} size={12} className="relative opacity-60 hover:opacity-100" />);
      
      code.parentNode?.insertBefore(buttonContainer, code.nextSibling);
    });

    // Add copy buttons to math equations
    const mathBlocks = container.querySelectorAll('.math-display, .math-inline');
    mathBlocks.forEach((math) => {
      if (math.querySelector('.copy-button-container')) return; // Already enhanced
      
      const text = getMathText(math as HTMLElement);
      if (!text.trim()) return;

      // Create container for the copy button
      const buttonContainer = document.createElement('div');
      
      if (math.classList.contains('math-display')) {
        // For display math, position button inside the scrolling container
        buttonContainer.className = 'copy-button-container absolute top-2 right-2 z-10';
        const katexDisplay = math.querySelector('.katex-display');
        if (katexDisplay) {
          (katexDisplay as HTMLElement).style.position = 'relative';
          (katexDisplay as HTMLElement).classList.add('group');
          katexDisplay.appendChild(buttonContainer);
        } else {
          // Fallback to original behavior
          (math as HTMLElement).style.position = 'relative';
          math.classList.add('group');
          math.appendChild(buttonContainer);
        }
      } else {
        // For inline math, position inline
        buttonContainer.className = 'copy-button-container inline-block ml-1 align-middle';
        math.classList.add('group', 'relative');
        math.parentNode?.insertBefore(buttonContainer, math.nextSibling);
      }
      
      // Create React root and render the copy button
      const root = createRoot(buttonContainer);
      const latex = text.startsWith('$') ? text : `$${text}$`; // Ensure LaTeX delimiters
      root.render(<CopyButton text={latex} size={12} className="relative" />);
    });

    // Cleanup function to unmount React components
    return () => {
      const copyContainers = container.querySelectorAll('.copy-button-container');
      copyContainers.forEach((container) => {
        const reactRoot = (container as any)._reactInternalFiber || (container as any)._reactInternalInstance;
        if (reactRoot) {
          try {
            createRoot(container).unmount();
          } catch (e) {
            // Ignore unmount errors
          }
        }
        container.remove();
      });
    };
  }, [htmlContent]);

  return contentRef;
} 