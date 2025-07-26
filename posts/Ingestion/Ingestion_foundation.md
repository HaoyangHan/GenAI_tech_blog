---
title: "Building a Production-Ready, Asynchronous Ingestion Pipeline for Complex Financial Documents"
category: "Technical Deep Dive"
date: "August 2, 2025"
summary: "The success of any RAG system hinges on the quality of its data ingestion. This article details the architecture of TinyRAG's ingestion pipeline, covering how we handle complex sources like PDFs and tables, our advanced chunking strategies, rich metadata extraction, and the dual-storage system using MongoDB and ChromaDB that powers our retrieval."
slug: "rag-ingestion-pipeline-for-financial-documents"
tags: ["RAG", "Ingestion", "Data Pipeline", "LlamaIndex", "ChromaDB", "MongoDB", "PyMuPDF", "Embedding"]
author: "Haoyang Han"
---

## Introduction: The Foundation of Insight

In our previous post, we established the "why" behind **TinyRAG**, detailing the **USD 156 million** business case for an agentic RAG system in financial memo generation. Now, we shift our focus to the "how," beginning with the most critical and foundational layer of the entire system: the **ingestion pipeline**.

The quality of a Retrieval Augmented Generation (RAG) system's output is a direct function of the data it ingests. Garbage in, garbage out. In the financial domain, "input" is a complex beast. We aren't dealing with simple text files. Our sources are a mix of:

*   **Unstructured Text:** Dense, multi-page PDFs like 10-K and 10-Q filings.
*   **Semi-structured Data:** Credit reports with tables and key-value pairs.
*   **Structured Data:** Financial models from spreadsheets.
*   **Rich Media:** Investor day presentations (`.pptx`) containing text, images, and charts.

A robust ingestion pipeline must do more than just read files. It must intelligently parse, dissect, enrich, and store this varied data in a way that maximizes retrieval relevance. This article breaks down the architecture of our production-ready, asynchronous ingestion pipeline.

<Image 
  src="/images/ingestion/llama_index_cheatsheet.png"
  alt="LlamaIndex Ingestion Pipeline Cheatsheet"
  width={1200}
  height={675}
  className="rounded-lg"
/>


## The Ingestion Workflow: From Raw File to Searchable Chunk

Our pipeline is designed as a series of specialized steps, each handling a distinct part of the process.

### 1. Document Loading: Selecting the Right Tool for the Job

The first step is to load the raw document into memory. For the prevalent PDF format, we use the powerful and lightweight [**PyMuPDF**](https://github.com/pymupdf/PyMuPDF) library. It's an ideal choice because it not only extracts text with high fidelity but also provides crucial metadata, such as the location (bounding boxes) of text blocks, images, and tables, which is essential for our special handling modules.

### 2. Chunking Strategy: Beyond Fixed-Size Slices

Once loaded, the document must be broken down into smaller pieces, or "chunks." Each chunk is a unit of text that will be converted into a numerical vector (embedding) for similarity search.

*   **Baseline Strategy:** The simplest method is fixed-size chunking. We start with a chunk size of `1024 tokens` and an overlap of `10%` (`~102 tokens`). The overlap ensures that semantic context isn't lost at the boundary between two chunks. While fast, this method can awkwardly split sentences or ideas.

*   **Advanced Strategy: Semantic Chunking:** For higher precision, we employ **semantic chunking**. As described in the [LlamaIndex documentation](https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/), this technique uses an LLM to identify semantic breaks in the text, creating chunks that are cohesive and contextually complete. For financial documents, where a single complex sentence can contain a critical insight, this preserves the integrity of the information and leads to far better retrieval results.

### 3. Metadata Extraction: Enriching Every Chunk

A raw text chunk has limited utility. To enable advanced filtering and improve context, we enrich every single chunk with a layer of metadata. This is achieved with a dedicated LLM call that extracts key information and returns a structured JSON object.

Here is the prompt template we designed for this task:

```json
{
  "prompt": "You are a data extraction expert. Analyze the following text chunk from a financial document. Extract the specified metadata. Respond ONLY with a valid JSON object. Do not include any explanatory text before or after the JSON.",
  "context": "Text Chunk: '{chunk_text}'",
  "output_format": {
    "keywords": ["list of 3-5 most relevant keywords"],
    "summary": "A concise, one-sentence summary of the chunk's core content.",
    "inferred_date": "YYYY-MM-DD or null if not present",
    "chunk_quality_score": "A score from 1 (low) to 5 (high) based on the clarity, specificity, and information density of the text. Low-quality text might be boilerplate, repetitive, or poorly formatted.",
    "entities": {
      "companies": ["list of company names"],
      "people": ["list of people's names"],
      "products": ["list of product names"]
    }
  }
}
```

This metadata is stored alongside the chunk and is invaluable during retrieval. An analyst can now query not just for text, but for "chunks with a quality score > 4 from Q4 2024 that mention 'revenue growth'."

### 4. Handling Special Formats: Tables and Images

Financial documents are not just walls of text. Tables and images carry some of the most vital information. We treat them as special chunk types.

*   **Tables:**
    1.  **Detection:** During the loading phase, `PyMuPDF` helps identify the boundaries of tables within a PDF.
    2.  **Extraction & Conversion:** The raw text of the table is extracted. If the source is a spreadsheet (e.g., CSV), we convert it directly to Markdown format.
    3.  **Understanding:** The raw table data is then passed to an LLM with a specialized prompt to generate a human-readable summary.
        > **Prompt:** "You are a financial analyst AI. Analyze the following table extracted from a financial document. Provide a concise, bulleted summary of the key trends, significant figures, and any notable anomalies or patterns. \n\nTable Data:\n\n{table_data_in_markdown}"
    4.  The resulting summary—not the raw table—becomes the text of the chunk. This creates a dense, insight-rich chunk that is highly effective for retrieval.

*   **Images, Charts, and Image-based PDFs:**
    *   **Baseline Method (OCR):** The simplest approach is to use Optical Character Recognition (OCR) to convert the image to text. The entire page's OCR output becomes a single chunk. This is fast but prone to errors.
    *   **Advanced Method (Multimodal Models):** For higher value, we leverage multimodal models like **GPT-4o** or **Gemini 2.5 Pro**. These models don't just "read" the text in an image; they *understand* it. They can interpret charts, describe trends in graphs, and explain complex diagrams. The rich, descriptive text generated by the multimodal model becomes the content of the chunk, capturing information that OCR would completely miss.

### 5. Embedding Models: The Engine of Similarity

With our chunks prepared, we convert them into vector embeddings. Our strategy involves a primary embedding model and a reranker for added precision.

*   **Embedding Models:**
    *   **`BAAI/bge-large-en`**: Our primary workhorse for high-performance English-language embeddings.
    *   **`BAAI/bge-multilingual-gemma2`**: Used for documents from international markets to ensure global coverage.
    *   **Qwen3 Embedding Model**: A strong alternative that we actively benchmark for performance.

*   **Reranker Model:**
    *   **`BAAI/bge-reranker-large`**: After an initial, broad retrieval from the vector store, a reranker model takes the top ~50 results and re-orders them based on a more nuanced understanding of semantic relevance. This two-stage process significantly improves the quality of the final context sent to the LLM.

### 6. Storage: A Dual-Database Architecture

A scalable RAG system requires a sophisticated storage strategy. We use a dual-database approach to handle different data needs.

*   **ChromaDB (The Vector Store):** This is where our chunks, their embeddings, and their extracted metadata live. ChromaDB is a specialized vector database built for one thing: extremely fast and efficient similarity search. We use the [LlamaIndex ChromaDB integration](https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo/) for seamless connection. This database answers the question: "Find me the chunks most similar to this query."

*   **MongoDB (The Document & Metadata Store):** This is our system's source of truth. Using **Beanie**, an asynchronous ODM for MongoDB, we define a clear schema to store the original documents and manage their lifecycle. This database answers questions like: "What is the processing status of `report_q4.pdf`?" or "Retrieve the original document and page number for this chunk citation."

Here is the Pydantic/Beanie model that defines our document structure in MongoDB:

```python
from datetime import datetime
from typing import List, Optional
from beanie import Document, Indexed
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    """Metadata for uploaded documents."""
    filename: str
    content_type: str
    size: int
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = False
    error: Optional[str] = None
    content_hash: Optional[str] = None  # SHA256 hash for duplicate detection

class DocumentChunk(BaseModel):
    """A chunk of text from a document, stored within the main document for reference."""
    text: str
    page_number: int
    chunk_index: int
    # Note: The embedding vector itself is stored in ChromaDB, not here.

class FinancialDocument(Document):
    """Document model for storing uploaded financial documents and their metadata."""
    user_id: Indexed(str)
    project_id: Indexed(str)
    
    # Core fields for tracking and validation
    filename: str
    content_type: str
    file_size: int
    status: str = "processing" # e.g., "processing", "completed", "failed"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Nested data
    metadata: DocumentMetadata
    chunks: List[DocumentChunk] = [] # Stores chunk text for reference and citation
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_deleted: bool = False

    class Settings:
        name = "financial_documents"
```

This dual-storage architecture gives us the best of both worlds: lightning-fast vector search from ChromaDB and the robust, scalable document management capabilities of MongoDB.

## Conclusion

The ingestion pipeline is the unsung hero of the TinyRAG system. By implementing an intelligent, multi-stage process—from semantic chunking to specialized handling of tables and images, rich metadata extraction, and a robust dual-storage backend—we build the high-quality foundation required for accurate and insightful generation. This meticulous preparation is what allows the retrieval system to find the true "alpha" hidden within the data.

In our next article, we'll build on this foundation and dive deep into the retrieval process itself. We will explore **<u>"Advanced Retrieval and Reranking Strategies for High-Stakes Q&A,"</u>** detailing how we construct complex queries and leverage our reranking models to deliver the most relevant context to the LLM.