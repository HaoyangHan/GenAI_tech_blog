-----
title: "TinyRAG Engineering Deep Dive: Your Questions Answered"
category: "Technical Deep Dive"
date: "July 27, 2025"
summary: "A comprehensive Q&A follow-up addressing key engineering decisions in TinyRAG's ingestion pipeline, including chunk size optimization, performance metrics, database selection trade-offs, metadata extraction strategies, and asynchronous processing architecture using Dramatiq and Redis."
slug: "tinyrag-engineering-deep-dive-qa"
tags: ["RAG", "Engineering", "Architecture", "ChromaDB", "Redis", "Dramatiq", "Vector Database", "Performance Optimization", "Async Processing", "Metadata Extraction"]
author: "Haoyang Han"

-----

In our last post, [**"Building a Production-Ready, Asynchronous Ingestion Pipeline for Complex Financial Documents,"**](https://www.google.com/search?q=https://tinybird.co/blog/rag-ingestion-pipeline-for-financial-documents) we detailed the core architecture of TinyRAG's data foundation. The response was fantastic, and it sparked a number of excellent, specific questions about our engineering decisions.

Welcome to the follow-up. This post is a technical Q\&A, where we'll reverse-engineer some of our key choices and provide the "why" behind the "how."

## 1\. Why Did You Start with a 1024-Token Chunk Size? 🤔

This is a great question about balancing theory and practice. The `1024`-token chunk size wasn't arbitrary; it was the result of **reverse-engineering our retrieval budget**.

In a RAG system, the final prompt sent to the Language Model (LLM) is a combination of the user's query, a system prompt, and the retrieved context (our chunks). This entire package must fit within the model's context window. We designed for a common and cost-effective context window size of `16k` tokens.

Our "context budget" calculation looked like this:

  * **System Prompt:** Our prompts are complex, guiding the model on persona, tone, and format. We allocate a generous `~2000` tokens for this.
  * **Retrieved Chunks:** We typically retrieve the top 8 most relevant chunks to provide sufficient context.
  * **LLM Response:** We need to leave room for the model to generate a detailed answer, typically `~400` to `~1200` tokens.

The math works out as follows:
$$\text{Total Context} = (\text{Num Chunks} \times \text{Chunk Size}) + \text{Prompt Size} + \text{Response Buffer}$$
$$16384 \approx (8 \times 1024) + 2000 + (400 \text{ to } 1200)$$
$$16384 \approx 8192 + 2000 + \text{Response} \approx 10192 + \text{Response}$$

This leaves ample room for the model's output. While newer models boast massive context windows, designing for a common denominator like `16k` creates a robust and widely compatible system.

  * **OpenAI:** `gpt-3.5-turbo-0125` (16k), `gpt-4o-mini` (128k)
  * **Meta:** `Llama-3.1-8B-Instruct` (128k)
  * **Google:** `Gemini 1.5 Flash` (1M)

While we now use more advanced methods like semantic chunking, the `1024`-token fixed-size slice was a calculated and effective starting point.

```python
# A simple representation of our context budget planning
def check_context_fit(
    num_chunks: int = 8,
    chunk_size: int = 1024,
    prompt_size: int = 2000,
    model_context_window: int = 16384
) -> bool:
    """Checks if the planned retrieval payload fits within the model's context window."""
    
    total_retrieval_size = num_chunks * chunk_size
    total_prompt_size = total_retrieval_size + prompt_size
    
    print(f"Total prompt size (retrieval + system): {total_prompt_size} tokens")
    
    if total_prompt_size < model_context_window:
        print(f"Payload fits within the {model_context_window} token limit.")
        return True
    else:
        print(f"Warning: Payload exceeds the {model_context_window} token limit.")
        return False

# Running the check for our baseline design
check_context_fit()
# Output:
# Total prompt size (retrieval + system): 10192 tokens
# Payload fits within the 16384 token limit.
```

-----

## 2\. What Is Your Typical Workload and Performance? ⏱️

It's one thing to talk architecture, but another to discuss real-world performance. Here are our typical production metrics:

  * **Document Volume:** A standard analysis project involves **3 to 10 core documents** (e.g., 10-Ks, credit reports, investor presentations), with a maximum of around 20 documents.
  * **Document Size:** Documents range from a few pages to over 200, with an average of **\~30 pages**.
  * **Processing Time:** The end-to-end ingestion time for an average 30-page document—including PDF parsing, chunking, metadata extraction, embedding, and storage—is approximately **1 minute**.
  * **Language:** The pipeline is optimized for **English** by default, using embedding models like `BAAI/bge-large-en-v1.5`. For multi-language projects, we can swap in a more capable (and slightly slower) model like `BAAI/bge-m3`, which supports over 100 languages.

-----

## 3\. What's the Impact of Using ChromaDB vs. a Postgres-based Vector Store? 💾

We received several questions about why we chose a specialized vector database like ChromaDB instead of using a PostgreSQL extension like `pgvector`. It's a classic trade-off between specialization and unification.

Our internal testing showed that a full ingestion cycle using a Postgres-based solution was approximately **20% slower** than our ChromaDB implementation.

**The Architectural Reason:**

  * **ChromaDB** is a **purpose-built vector database**. Its entire architecture, from data storage formats to query execution engines, is optimized for one thing: extremely fast Approximate Nearest Neighbor (ANN) search using algorithms like HNSW (Hierarchical Navigable Small World). It's designed for retrieval-heavy workloads where vector search speed is the primary concern.

  * **PostgreSQL with `pgvector`** is a **general-purpose relational database** with an added capability. This is powerful because you can keep your structured metadata and vectors in the same database, simplifying your stack and enabling transactional integrity (ACID compliance). However, its indexing and query planning are not solely focused on vector operations. While highly capable, it often can't match the raw query-per-second performance of a specialized system like ChromaDB at scale.

For TinyRAG, where the quality of retrieval is paramount and the system is constantly performing similarity searches, the performance gain from a specialized vector store was worth the added complexity of a dual-database system.

```python
# In our previous post, we showed the ChromaDB setup.
# Here's a hypothetical setup for pgvector with LlamaIndex
# to illustrate the alternative.

# from llama_index.vector_stores.postgres import PGVectorStore
# import psycopg2

# # Connection to a PostgreSQL database
# db_name = "tiny_rag_db"
# host = "localhost"
# password = "your_password"
# port = "5432"
# user = "postgres"

# # Establish connection and cursor
# conn = psycopg2.connect(
#     dbname="postgres", host=host, password=password, port=port, user=user
# )
# conn.autocommit = True

# # NOTE: This is a simplified example. A production setup requires more configuration.
# vector_store_postgres = PGVectorStore.from_params(
#     database=db_name,
#     host=host,
#     password=password,
#     port=port,
#     user=user,
#     table_name="financial_docs_embeddings",
#     embed_dim=1024,  # Must match the dimension of your embedding model (e.g., bge-large is 1024)
# )

# # The storage_context would then use this vector_store instance
# # storage_context = StorageContext.from_defaults(vector_store=vector_store_postgres)
```

-----

## 4\. What If Automated Metadata Extraction Fails? 🏷️

Our pipeline uses an LLM to extract critical metadata like `inferred_date` and `document_type`. But what if the text is ambiguous and the LLM can't confidently make an extraction?

This is where a **human-in-the-loop** design becomes crucial.

If our Pydantic model validation fails to receive a required field like a date or document type, the process doesn't halt. Instead, the document is flagged in our system. The UI then **forces the user to manually select the correct values** from a dropdown before the document can be included in a project.

**The Impact:** Chunks from a document with missing or unconfirmed metadata are not given any preferential treatment or filtering capabilities during retrieval. Forcing the user to provide this information ensures that our advanced filtering logic ("*find risks mentioned in Q4 2024 reports*") remains reliable. It's a simple, robust solution to an unavoidable problem.

-----

## 5\. Can You Detail the Asynchronous Processing Architecture? ⚙️

Processing hundreds of chunks for a single document can be slow if done sequentially. We designed our system to be asynchronous from the ground up using a powerful trio of technologies: **Dramatiq**, **Redis**, and **`asyncio`**.

  * **Redis:** An in-memory data store that acts as a lightning-fast **message broker**. When a task needs to be run, it's sent as a message to a Redis queue.
  * **Dramatiq:** A reliable, simple **distributed task queue** library for Python. It defines "actors"—functions that can be executed in the background by separate worker processes. Dramatiq handles the logistics of sending tasks to the Redis queue and ensuring they get processed.
  * **`asyncio`:** Python's native library for writing asynchronous code, used *within* each Dramatiq worker to perform non-blocking I/O operations (like making API calls to an LLM or database) concurrently.

Here's a simplified look at how we define a chunk processing task as a Dramatiq actor:

```python
import dramatiq
import os
from dramatiq.brokers.redis import RedisBroker

# In a real app, this is configured once at startup
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_broker = RedisBroker(url=REDIS_URL)
dramatiq.set_broker(redis_broker)

# This is our "actor" - a function designed to be run in the background.
@dramatiq.actor
def process_chunk(document_id: str, chunk_index: int, chunk_text: str):
    """
    A single, atomic task to process one chunk of a document.
    This would be run by a separate worker process.
    """
    print(f"Worker processing chunk {chunk_index} for doc {document_id}...")

    # 1. Extract metadata (LLM call)
    # metadata = extract_metadata_for_chunk(chunk_text)
    
    # 2. Generate embedding (ML model call)
    # embedding = Settings.embed_model.get_text_embedding(chunk_text)

    # 3. Store the result in ChromaDB and update MongoDB
    # (These would be async database calls)
    
    # Simulate work
    import time
    time.sleep(2) 
    
    print(f"Finished processing chunk {chunk_index} for doc {document_id}.")
    # In a real scenario, we'd return data or update a database.
    return True

# --- How this actor is called from the main application ---
# This would happen after a document is uploaded and split into chunks.
def enqueue_document_processing(document_id: str, chunks: list):
    """
    Sends each chunk to the Redis queue for a worker to process.
    """
    for i, chunk_text in enumerate(chunks):
        # .send() does not run the function here. It just places a message
        # on the Redis queue for a Dramatiq worker to pick up.
        process_chunk.send(document_id, i, chunk_text)
    
    print(f"Enqueued {len(chunks)} processing tasks for document {document_id}.")

# Example Usage:
# chunks_from_pdf = ["Text of chunk 1...", "Text of chunk 2...", "Text of chunk 3..."]
# enqueue_document_processing("doc_xyz_123", chunks_from_pdf)

# To run this, you would start one or more Dramatiq workers in your terminal:
# $ dramatiq your_module_name
```

This architecture allows us to process dozens of chunks in parallel, dramatically reducing the total ingestion time.

## Conclusion

Building a production-ready RAG system is a game of deliberate trade-offs and thoughtful engineering. Every choice, from chunk size to database selection, has a cascading effect on performance, scalability, and retrieval quality. By sharing our reasoning, we hope to provide a clearer picture of what it takes to move from a simple RAG prototype to a robust, enterprise-grade system.

Stay tuned for our next article, where we'll finally move up the stack to explore **"Advanced Retrieval and Reranking Strategies for High-Stakes Q\&A."**