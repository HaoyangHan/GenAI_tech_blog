import Header from '@/components/layout/Header';
import { BookOpen, Brain, Code, Lightbulb } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      <main className="max-w-4xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            Welcome to My GenAI Journey
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            A personal exploration into the fascinating world of Generative AI, where I share my learnings, 
            insights, and discoveries as I navigate through the rapidly evolving landscape of artificial intelligence.
          </p>
        </div>

        {/* Personal Introduction */}
        <div className="mb-16">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-8 mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-6">About Me</h2>
            <div className="prose prose-lg max-w-none text-gray-700">
              <p className="mb-4">
                Hi, I'm <strong>Haoyang Han</strong>, a passionate technologist fascinated by the transformative power of 
                Generative AI. My journey began with curiosity about how machines could create, understand, and reason 
                like humans, and has evolved into a deep dive into the technical foundations that make it all possible.
              </p>
              <p className="mb-4">
                Through this blog, I document my exploration of everything from fundamental mathematical concepts 
                like attention mechanisms and transformers, to practical implementations of RAG systems, 
                prompt engineering techniques, and agentic workflows.
              </p>
              <p>
                My goal is to bridge the gap between theoretical understanding and practical application, 
                sharing both the "what" and the "why" behind the technologies that are reshaping our world.
              </p>
            </div>
          </div>
        </div>

        {/* What You'll Find Here */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">What You'll Find Here</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Brain className="w-8 h-8 text-blue-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Deep Technical Insights</h3>
              <p className="text-gray-600 text-sm">
                Mathematical foundations, architecture deep-dives, and implementation details
              </p>
            </div>
            
            <div className="text-center">
              <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Code className="w-8 h-8 text-green-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Practical Implementations</h3>
              <p className="text-gray-600 text-sm">
                Code examples, tutorials, and hands-on guides for building AI systems
              </p>
            </div>
            
            <div className="text-center">
              <div className="bg-purple-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Lightbulb className="w-8 h-8 text-purple-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Strategic Thinking</h3>
              <p className="text-gray-600 text-sm">
                Business applications, evaluation frameworks, and future trends
              </p>
            </div>
            
            <div className="text-center">
              <div className="bg-orange-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <BookOpen className="w-8 h-8 text-orange-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Learning Resources</h3>
              <p className="text-gray-600 text-sm">
                Curated papers, tools, and resources for continuous learning
              </p>
            </div>
          </div>
        </div>

        {/* My Philosophy */}
        <div className="bg-gray-50 rounded-2xl p-8 mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-6">My Philosophy</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">🎯 Depth Over Breadth</h3>
              <p className="text-gray-600">
                Rather than skimming the surface of many topics, I prefer to dive deep into 
                fundamental concepts and truly understand how things work.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">🔬 Theory Meets Practice</h3>
              <p className="text-gray-600">
                Every theoretical concept should have practical applications. I strive to 
                connect mathematical foundations with real-world implementations.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">🌱 Continuous Learning</h3>
              <p className="text-gray-600">
                The field of AI evolves rapidly. I believe in documenting the learning 
                journey to help both myself and others grow.
              </p>
            </div>
          </div>
        </div>

        {/* Connect */}
        <div className="text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-6">Let's Connect</h2>
          <p className="text-lg text-gray-600 mb-8">
            I'm always excited to discuss AI, share insights, and learn from fellow enthusiasts.
          </p>
          <div className="space-y-4">
            <p className="text-gray-600">
              Feel free to reach out if you have questions, want to collaborate, or just want to chat about 
              the fascinating world of Generative AI.
            </p>
            <p className="text-sm text-gray-500">
              This blog is built with Next.js and supports advanced markdown features including 
              LaTeX equations, syntax highlighting, and interactive elements.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
} 