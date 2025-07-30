declare global {
  interface Window {
    dataLayer: any[];
    gtag: (
      command: 'config' | 'set' | 'event',
      targetId: string,
      config?: any
    ) => void;
  }
}

export {}; 