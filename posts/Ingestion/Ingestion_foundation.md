---
title: "Building a Production-Ready, Asynchronous Ingestion Pipeline for Complex Financial Documents"
category: "Technical Deep Dive"
date: "July 27, 2025"
summary: "The success of any RAG system hinges on the quality of its data ingestion. This article details the architecture of TinyRAG's ingestion pipeline, covering how we handle complex sources like PDFs and tables, our advanced chunking strategies, rich metadata extraction, and the dual-storage system using MongoDB and ChromaDB that powers our retrieval."
slug: "rag-ingestion-pipeline-for-financial-documents"
tags: ["RAG", "Ingestion", "Data Pipeline", "LlamaIndex", "ChromaDB", "MongoDB", "PyMuPDF", "Embedding"]
author: "Haoyang Han"
---

**📚 Next Article in Series:**  
→ [TinyRAG Ingestion Deep Dive: Your Questions Answered](/post/tinyrag-engineering-deep-dive-qa)


## Introduction: The Foundation of Insight

In our previous post, we established the "why" behind **TinyRAG**, detailing the **USD 156 million** business case for an agentic RAG system in financial memo generation. Now, we shift our focus to the "how," beginning with the most critical and foundational layer of the entire system: the **ingestion pipeline**.

The quality of a Retrieval Augmented Generation (RAG) system's output is a direct function of the data it ingests. Garbage in, garbage out. In the financial domain, "input" is a complex beast. We aren't dealing with simple text files. Our sources are a mix of:

*   **Unstructured Text:** Dense, multi-page PDFs like 10-K and 10-Q filings.
*   **Semi-structured Data:** Credit reports with tables and key-value pairs.
*   **Structured Data:** Financial models from spreadsheets.
*   **Rich Media:** Investor day presentations (`.pptx`) containing text, images, and charts.

A robust ingestion pipeline must do more than just read files. It must intelligently parse, dissect, enrich, and store this varied data in a way that maximizes retrieval relevance. This article breaks down the architecture of our production-ready, asynchronous ingestion pipeline with real code examples.

<Image 
  src="/images/ingestion/llama_index_cheatsheet.png"
  alt="LlamaIndex Ingestion Pipeline Cheatsheet"
  width={1200}
  height={675}
  className="rounded-lg"
/>
---

## The Ingestion Workflow: From Raw File to Searchable Chunk

Our pipeline is designed as a series of specialized, asynchronous steps. At a high level, the journey of a single document looks like this:

1.  **Load:** A raw file (e.g., PDF) is loaded into a structured text format.
2.  **Chunk:** The text is broken into small, semantically meaningful pieces.
3.  **Enrich:** Each chunk is enhanced with metadata (keywords, summaries, etc.) and special processing is applied for tables and images.
4.  **Embed & Store:** The enriched chunks are converted into vector embeddings and stored in a specialized vector database (ChromaDB), while the original document's metadata is tracked in a document database (MongoDB).

Let's dive into the engineering decisions behind each step.

### 1. Document Loading: How to Reliably Extract Text from Complex PDFs?

Financial reports in PDF format are notoriously difficult to parse. They contain complex layouts with columns, headers, footers, tables, and images. A simple text extraction can fail spectacularly. We need a tool that understands document structure.

For this, we use [**PyMuPDF**](https://github.com/pymupdf/PyMuPDF). It's a high-performance library that not only extracts text with high fidelity but also provides crucial structural information, such as the location (bounding boxes) of text blocks, images, and tables. This is essential for our downstream processing.

Here's a function to read text from a PDF, keeping it organized by page.

```python
import fitz # PyMuPDF
from typing import List

def load_pdf_text(file_path: str) -> List[dict]:
    """
    Loads text from each page of a PDF and returns it as a list of dictionaries.
    
    Args:
        file_path: The path to the PDF file.
        
    Returns:
        A list of dictionaries, where each dictionary contains the page number 
        and the extracted text content.
    """
    doc = fitz.open(file_path)
    pages_content = []
    for page_num, page in enumerate(doc):
        pages_content.append({
            "page_number": page_num + 1,
            "content": page.get_text()
        })
    doc.close()
    return pages_content

# Example usage:
# pages = load_pdf_text("data/sample-10k-report.pdf")
# print(f"Loaded {len(pages)} pages.")
# print(f"Content from page 1: {pages[0]['content'][:300]}...")
```

---

### 2. Chunking Strategy: How to Break Down Documents Without Losing Meaning?

Once loaded, the text must be broken into "chunks." This decision is critical: if a chunk splits a key idea in half, retrieval will fail.

#### Baseline Strategy: Fixed-Size Slicing

The simplest approach is fixed-size chunking. We define a chunk size (e.g., `1024 tokens`) and an overlap to maintain some context between chunks. It's fast but naive.

```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

# Assume 'pages' is the list of dicts from the previous step
# We create LlamaIndex Document objects for processing
llama_documents = [
    Document(text=page['content'], metadata={"page_label": page['page_number']}) 
    for page in pages
]

# Baseline chunking
splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=102, # ~10% overlap
)

nodes = splitter.get_nodes_from_documents(llama_documents)
print(f"Split document into {len(nodes)} chunks (nodes).")
```

#### Advanced Strategy: Semantic Chunking

A far better approach for our high-stakes use case is **semantic chunking**. As detailed in the [LlamaIndex documentation](https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/), this technique uses an embedding model to find natural semantic breaks in the text. This creates chunks that are contextually whole, even if it means they have variable lengths. The trade-off is a slower processing time (due to LLM calls), but the resulting retrieval quality is significantly higher.

```python
import os
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

# Note: This requires an OpenAI API key for the embedding model
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

# The SemanticSplitter uses an embedding model to find semantic breaks
semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1, # Number of sentences to group on either side of a breakpoint
    breakpoint_percentile_threshold=95, # The percentile of similarity scores to use as a breakpoint
    embed_model=OpenAIEmbedding()
)

# This process is slower and more expensive but yields higher quality chunks
semantic_nodes = semantic_splitter.get_nodes_from_documents(llama_documents)
print(f"Split document into {len(semantic_nodes)} semantic chunks.")
```

---

### 3. Metadata Extraction: How to Make Raw Chunks Searchable and Smart?

A raw text chunk is "dumb." To enable advanced, filtered queries ("*find discussions of 'liquidity risk' from Q4 2024*"), we must enrich each chunk with structured metadata. We use a dedicated LLM call with a strictly defined output schema to achieve this.

The `instructor` library is perfect for this, as it forces the LLM's output to conform to a Pydantic model, ensuring reliable, structured data.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

# This enables response validation and parsing against Pydantic models
client = instructor.from_openai(OpenAI())

# Define the structured output using Pydantic
class Entities(BaseModel):
    companies: Optional[List[str]] = Field(default_factory=list)
    people: Optional[List[str]] = Field(default_factory=list)
    products: Optional[List[str]] = Field(default_factory=list)

class ChunkMetadata(BaseModel):
    keywords: List[str] = Field(..., description="List of 3-5 most relevant keywords.")
    summary: str = Field(..., description="A concise, one-sentence summary.")
    inferred_date: Optional[str] = Field(None, description="YYYY-MM-DD or null.")
    chunk_quality_score: int = Field(..., ge=1, le=5, description="Score from 1-5 for text quality.")
    entities: Entities

def extract_metadata_for_chunk(chunk_text: str) -> ChunkMetadata:
    """
    Uses an LLM to extract structured metadata from a text chunk.
    """
    try:
        metadata = client.chat.completions.create(
            model="gpt-4o",
            response_model=ChunkMetadata,
            messages=[
                {"role": "system", "content": "You are a data extraction expert. Analyze the following text chunk from a financial document. Extract the specified metadata. Respond ONLY with the structured data."},
                {"role": "user", "content": f"Text Chunk: '{chunk_text}'"}
            ]
        )
        return metadata
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return None

# Example with a node from our chunking step
# first_chunk_text = nodes[0].get_content()
# extracted_meta = extract_metadata_for_chunk(first_chunk_text)
# if extracted_meta:
#    print(extracted_meta.model_dump_json(indent=2))
```

---

### 4. Special Formats: How to Handle Data That Isn't Plain Text?

Financial documents are rich with tables and images. Simply running OCR over them is insufficient as it loses all structural context. We must treat them as first-class citizens in our pipeline.

#### Tables: Summarize, Don't Just Extract

Instead of embedding a noisy, raw table string, we extract the table, convert it to a clean format like Markdown, and then use an LLM to **generate a human-readable summary**. This summary—dense with insights about trends and key figures—is what gets embedded.

```python
import fitz # PyMuPDF
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def summarize_table(table_markdown: str) -> str:
    """Uses an LLM to generate a summary of a table."""
    prompt = f"""
    You are a financial analyst AI. Analyze the following table extracted from a financial document.
    Provide a concise, bulleted summary of the key trends, significant figures, and any notable anomalies or patterns.
    
    Table Data (in Markdown format):
    
    {table_markdown}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

def process_document_tables(file_path: str):
    """
    Finds tables in a PDF, converts them to Markdown, and generates a summary for each.
    These summaries can then be added as distinct nodes/chunks.
    """
    doc = fitz.open(file_path)
    table_summaries = []
    for page in doc:
        tables = page.find_tables()
        if tables:
            print(f"Found {len(tables)} table(s) on page {page.number + 1}")
            for i, table in enumerate(tables):
                table_md = table.to_markdown(clean=True)
                summary = summarize_table(table_md)
                table_summaries.append({
                    "page_number": page.number + 1,
                    "table_index": i,
                    "summary": summary
                })
    return table_summaries

# table_nodes_data = process_document_tables("data/sample-10k-report.pdf")
# if table_nodes_data:
#     print(table_nodes_data[0]['summary'])
```

#### Images and Charts: From Pixels to Insights

For images, we contrast a baseline OCR approach with an advanced multimodal one.

*   **Baseline (OCR):** Fast but misses visual context.
*   **Advanced (Multimodal LLM):** Models like **Gemini 1.5 Pro** can *interpret* a chart, describing its trends, axes, and meaning. This rich description is far more valuable for retrieval than raw OCR text.

```python
import google.generativeai as genai
from PIL import Image

# Configure the Gemini API
# genai.configure(api_key="YOUR_GEMINI_API_KEY")

def describe_image_with_gemini(image_path: str, prompt: str) -> str:
    """Uses a multimodal model to describe an image."""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        img = Image.open(image_path)
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        return f"Error with multimodal model: {e}"

# chart_prompt = "Describe this financial chart. What are the key trends, axes, and important data points?"
# description = describe_image_with_gemini("data/chart.png", chart_prompt)
# print(description)
```

---

### 5. Embedding & Reranking: How to Ensure We Find the Most Relevant Chunks?

Finding the right information is a two-step process:

1.  **Embedding:** We convert our enriched chunks into numerical vectors using a powerful open-source model like `BAAI/bge-large-en-v1.5`. This allows for a fast, broad similarity search.
2.  **Reranking:** The initial search might return semantically related but not perfectly relevant chunks. A **reranker** model (`BAAI/bge-reranker-large`) takes the top results from the initial search and re-orders them based on a more nuanced understanding of the query's intent, dramatically improving precision.

Here’s how to configure this two-stage retrieval pipeline in `LlamaIndex`:

```python
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

# 1. Configure the primary embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

# 2. Configure the reranker model
reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-large", 
    top_n=5 # Return the top 5 most relevant results after reranking
)

# Assume 'nodes' and 'storage_context' (from next section) are defined
# index = VectorStoreIndex(nodes, storage_context=storage_context)

# 3. Create a query engine with the reranker
query_engine = index.as_query_engine(
    similarity_top_k=20,  # Retrieve 20 initial candidates from the vector store
    node_postprocessors=[reranker] # Apply the reranker to the candidates for a final, precise list
)

# 4. Execute a query and see the reranked results
# response = query_engine.query("What were the main drivers of revenue growth in the last quarter?")
# for node in response.source_nodes:
#     print(f"Reranked Score: {node.score:.4f}, Page: {node.metadata.get('page_label')}")
```

---

### 6. Storage: How to Manage Both Vector and Document Data at Scale?

A production RAG system has two distinct storage needs, leading us to a dual-database architecture.

#### ChromaDB: The Vector Store

For the task of finding the most similar chunks, we need a database optimized for high-speed vector search. [**ChromaDB**](https://www.trychroma.com/) is an open-source vector database perfect for this. We use the [LlamaIndex ChromaDB integration](https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo/) to store our embeddings and their associated metadata.

```python
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

# Assume 'nodes' contains our text chunks, now enriched with metadata
# for i, node in enumerate(nodes):
#     node.metadata = extract_metadata_for_chunk(node.get_content()).model_dump()

# 1. Initialize a ChromaDB client for persistent storage
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("financial_docs_v1")

# 2. Assign ChromaDB as the vector store in our storage context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 3. Create the index. This embeds the 'nodes' and stores them in ChromaDB.
index = VectorStoreIndex(nodes, storage_context=storage_context)
print("Embeddings have been successfully stored in ChromaDB.")
```

#### MongoDB: The Document Lifecycle & Metadata Store

While ChromaDB handles the search, we need a separate, robust database to be our system's source of truth. Here we track the original files, their processing status, user information, and references to their chunks. [**MongoDB**](https://www.mongodb.com/) is ideal for this flexible, metadata-heavy workload. We use [**Beanie**](https://github.com/roman-right/beanie), an asynchronous Object-Document Mapper (ODM), to define a clear schema and interact with the database efficiently.

```python
import motor.motor_asyncio
import asyncio
from datetime import datetime
from typing import List, Optional
from beanie import Document, Indexed, init_beanie
from pydantic import BaseModel, Field

# --- Pydantic and Beanie Models ---
class DocumentMetadata(BaseModel):
    filename: str
    content_type: str
    size: int
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = False
    error: Optional[str] = None
    content_hash: Optional[str] = None

class DocumentChunk(BaseModel):
    text: str
    page_number: int
    chunk_index: int
    # Note: The embedding vector itself is in ChromaDB. This is for reference.

class FinancialDocument(Document):
    user_id: Indexed(str)
    project_id: Indexed(str)
    filename: str
    content_type: str
    file_size: int
    status: str = "processing"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: DocumentMetadata
    chunks: List[DocumentChunk] = []
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_deleted: bool = False

    class Settings:
        name = "financial_documents"

# --- Asynchronous Initialization and Usage ---
async def manage_document_in_mongo():
    # In a real app, this connection is established once at startup
    client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
    await init_beanie(database=client.tiny_rag_db, document_models=[FinancialDocument])

    # 1. Create a new document record when a file is uploaded
    new_doc = FinancialDocument(
        user_id="analyst_123",
        project_id="case_abc",
        filename="report_q4.pdf",
        content_type="application/pdf",
        file_size=1572864,
        metadata=DocumentMetadata(
            filename="report_q4.pdf",
            content_type="application/pdf",
            size=1572864
        )
    )
    await new_doc.insert()
    print(f"Created document record with ID: {new_doc.id}")

    # 2. After processing is complete, update its status
    # (In a real pipeline, you'd fetch the document by its ID and update it)
    new_doc.status = "completed"
    new_doc.metadata.processed = True
    new_doc.updated_at = datetime.utcnow()
    await new_doc.save()
    print(f"Updated document {new_doc.id} to status: '{new_doc.status}'")

# if __name__ == "__main__":
#     asyncio.run(manage_document_in_mongo())
```

This dual-storage architecture gives us the best of both worlds: lightning-fast vector search from ChromaDB and robust, scalable document lifecycle management in MongoDB.

---

## Conclusion

The ingestion pipeline is the unsung hero of the TinyRAG system. By implementing an intelligent, multi-stage process—from semantic chunking to specialized handling of tables and images, rich metadata extraction, and a robust dual-storage backend—we build the high-quality foundation required for accurate and insightful generation. This meticulous preparation is what allows the retrieval system to find the true "alpha" hidden within the data.

In our next article, we'll build on this foundation and dive deep into the retrieval process itself. We will explore **<u>"Advanced Retrieval and Reranking Strategies for High-Stakes Q&A,"</u>** detailing how we construct complex queries and leverage our reranking models to deliver the most relevant context to the LLM.