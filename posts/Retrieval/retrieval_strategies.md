---
title: "Retrieval Strategies in Financial RAG Systems: From Dense to Hybrid Approaches"
category: "Retrieval"
date: "July 28, 2025"
summary: "A deep dive into retrieval strategies for production-grade RAG systems, comparing dense, sparse, and hybrid retrieval, and their impact on financial document QA. Learn how to select and tune retrieval for high-stakes, high-recall use cases."
slug: "retrieval-strategies-in-financial-rag"
tags: ["RAG", "Retrieval", "Dense Retrieval", "Hybrid Search", "Financial AI", "LlamaIndex", "Vector Search"]
author: "Haoyang Han"
---

**📚 RAG Implementation Series - Article 7 of 9:**  
[Complete Learning Path](/knowledge/rag) | ← [Previous: System Architecture](/post/architecting-production-ready-financial-rag-system) | **Current: Retrieval Strategies** → [Next: Evaluation Framework](/post/framework-rigorous-evaluation-agentic-postprocessing-financial-rag)


# Advanced Retrieval Architectures in LlamaIndex for Financial RAG Systems

## Introduction: The Quest for Precision in Financial RAG

The generation of financial memoranda, analyses, and reports using Large Language Models (LLMs) represents a significant leap forward in automating complex knowledge work. However, this domain demands a level of accuracy and reliability that far exceeds the requirements of general-purpose chatbots. In finance, errors, hallucinations, or the presentation of out-of-date information can lead to flawed decisions with material consequences. Consequently, the core of a high-performing financial Retrieval-Augmented Generation (RAG) system is not the generator model alone, but its retrieval pipeline—the sophisticated mechanism responsible for sourcing the precise, relevant, and timely information the LLM needs to formulate its response.

The evolution of RAG has moved decisively beyond naive, single-stage vector search. A modern, production-grade RAG architecture is a multi-stage process involving query transformation, hybrid retrieval strategies, and advanced reranking modules designed to progressively refine context and maximize relevance. This report provides a comprehensive technical guide to building such a pipeline, detailing five key strategies that form a layered approach to retrieval optimization.

Each strategy will be explored from its conceptual foundations to its practical implementation, leveraging the LlamaIndex framework. LlamaIndex is a powerful and modular data framework designed specifically for building LLM applications, chosen for its flexibility and extensive ecosystem of integrations that allow for the construction of customized and sophisticated data workflows. The following sections will provide data scientists and engineers with the principles, code, and analytical trade-offs necessary to construct a state-of-the-art retrieval system tailored to the exacting demands of financial analysis.

## Section I: Foundational Retrieval: Vector Search with Metadata-Driven Filtering

The bedrock of any modern RAG system is dense retrieval, which leverages the power of transformer-based embedding models to search for information based on semantic meaning rather than just keyword overlap. This initial stage is designed to cast a wide net, prioritizing the capture of all potentially relevant information—a principle known as maximizing recall.

### 1.1. Conceptual Framework: Dense Retrieval with BGE-large-en-v1.5

At the heart of dense retrieval are **bi-encoder** models. These models, such as the BAAI General Embedding (BGE) series, are designed to create a fixed-size numerical representation—an embedding—for any given piece of text. During the indexing phase, each document chunk is passed through the bi-encoder, and its resulting vector is stored in a specialized vector database. This process maps the entire corpus into a high-dimensional semantic space where texts with similar meanings are located closer to one another. At query time, the user's query is encoded using the same model, and retrieval becomes a highly efficient nearest-neighbor search within this space, typically measured using cosine similarity.

The choice of embedding model is critical. For a financial RAG system, a model like `BAAI/bge-large-en-v1.5` is an excellent starting point. It is part of a family of models specifically trained and optimized for retrieval-augmented LLM applications. A key advantage of the v1.5 series is its enhanced ability to perform retrieval without requiring a specific instruction prefix, which simplifies its integration into the pipeline. As a powerful, open-source model, it is well-suited to capturing the complex semantic relationships and nuanced terminology found in financial documents.

The core strategy for this foundational stage is to retrieve a large number of candidate chunks, for example, setting `similarity_top_k=100`. This approach intentionally favors *recall* (finding all relevant documents) over *precision* (ensuring all retrieved documents are relevant). The underlying assumption is that it is better to retrieve some irrelevant documents alongside the relevant ones than to miss a critical piece of information entirely. The task of filtering out the noise and improving precision is delegated to subsequent, more specialized stages of the pipeline, such as reranking.

### 1.2. Strategic Implementation: Code and Practice

Implementing this foundational retrieval stage in LlamaIndex involves setting up the embedding model, enriching the data with high-quality metadata, and then using that metadata to efficiently filter the search space before the vector search occurs.

#### Setup and Embedding Model Configuration

First, the environment is configured to use the specified BGE model. The `HuggingFaceEmbedding` class from LlamaIndex provides a seamless wrapper around models available on the Hugging Face Hub.

```python
import llama_index.core
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

# Set a consistent context for the application
# Using a powerful open-source embedding model optimized for retrieval
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5",
    trust_remote_code=True
)
Settings.llm = None # We are focusing only on retrieval for now
Settings.chunk_size = 512
Settings.chunk_overlap = 64

print("✅ Settings configured")

# In a real scenario, documents would be financial reports, filings, etc.
# We will simulate this with placeholder documents
# Ensure you have a 'data' directory with some text files.
documents = SimpleDirectoryReader("data").load_data()
print(f"✅ Loaded {len(documents)} document(s)")
````

#### Metadata Extraction and Filtering

For financial documents, metadata is invaluable. Relevant metadata fields could include `report_type` (e.g., '10-K', '10-Q', 'Earnings Call Transcript'), `fiscal_year`, `fiscal_quarter`, and `company_ticker`. During indexing, this metadata should be associated with each document chunk. LlamaIndex automatically propagates metadata from documents to their derived nodes (chunks).

When querying, we can apply a `MetadataFilter` to narrow the search space. This pre-filtering step is highly efficient as it reduces the number of vectors that need to be scored, improving both speed and relevance.

```python
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

# This function would be a sophisticated parser in a real system
# Here, we just add static metadata for demonstration
for doc in documents:
    doc.metadata = {
        "report_type": "10-K",
        "fiscal_year": 2023,
        "company_ticker": "TINYCORP"
    }

# Setup vector store (e.g., Qdrant)
client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="financial_memos")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create the index with the documents and metadata
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

print("✅ Index created with metadata")

# Build a retriever with metadata filters and a high top_k for recall
retriever = index.as_retriever(
    vector_store_query_mode="default",
    similarity_top_k=100,
    filters=MetadataFilters(
        filters=[
            ExactMatchFilter(key="company_ticker", value="TINYCORP"),
            ExactMatchFilter(key="fiscal_year", value=2023),
        ]
    ),
)

query = "What were the primary risk factors for TINYCORP in fiscal year 2023?"
retrieved_nodes = retriever.retrieve(query)

print(f"Retrieved {len(retrieved_nodes)} nodes using vector search + metadata filters.")
# These 100 nodes will now be passed to a reranker.
```

### 1.3. Analysis and Trade-offs

  * **Pros:**

      * **High Recall:** Retrieving a large `top_k` minimizes the risk of missing relevant context.
      * **Semantic Power:** Captures the meaning behind financial jargon and complex queries that keyword search would miss.
      * **Efficiency at Scale:** Metadata pre-filtering significantly prunes the search space, making the process faster and more cost-effective.

  * **Cons:**

      * **Low Precision:** The initial set of 100 chunks will inevitably contain noise and less relevant information.
      * **Context Stuffing:** Passing all 100 chunks directly to an LLM would result in a bloated, noisy prompt, potentially confusing the model and exceeding context window limits.
      * **Bi-Encoder Limitation:** While powerful, bi-encoders score each document independently of the query, which can be less precise than models that consider the query and document together.

## Section II: Lexical Refinement with BM25

While dense retrieval excels at understanding semantic meaning, it can sometimes overlook documents that contain exact keyword matches, especially for specific, technical terms, company names, or financial metrics. To mitigate this, we can introduce a lexical search algorithm like Okapi BM25.

### 2.1. Conceptual Framework: The Power of Term Statistics

Okapi BM25 (Best Match 25) is a sparse retrieval algorithm that ranks documents based on the query terms they contain. Unlike simple keyword counting, BM25 is more sophisticated, incorporating two key principles:

1.  **Term Frequency Saturation:** The relevance of a term does not grow infinitely with its frequency. The first few mentions of a keyword are highly significant, but subsequent mentions provide diminishing returns. BM25 models this by using a saturation function.
2.  **Inverse Document Frequency (IDF):** Terms that appear in many documents across the corpus (e.g., "company," "revenue") are less informative than rare terms (e.g., "amortization of intangible assets"). BM25 penalizes common terms and boosts the importance of rare ones.

The scoring formula for a query $Q$ with terms $q\_1, ..., q\_n$ for a document $D$ is:

$$
\text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}
$$Where:

* $f(q\_i, D)$ is the term frequency of $q\_i$ in document $D$.
* $|D|$ is the length of document $D$.
* $\\text{avgdl}$ is the average document length in the corpus.
* $k\_1$ and $b$ are hyperparameters that control term frequency saturation and document length normalization, respectively.

By combining BM25 with vector search (a "hybrid" approach), we get the best of both worlds: the semantic understanding of dense retrieval and the keyword precision of sparse retrieval.

### 2.2. Strategic Implementation: Hybrid Retrieval in LlamaIndex

In LlamaIndex, BM25 can be used as a standalone retriever or, more powerfully, as part of a reranking or fusion process. For our pipeline, we will use it as a reranker to refine the initial 100 chunks retrieved by the vector search.

```python
from llama_index.core.retrievers import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank

# Assume 'retrieved_nodes' contains the 100 nodes from the previous step
# We need the full document corpus to initialize the BM25Retriever
# For simplicity, let's re-use the 'documents' loaded earlier

# The BM25Retriever needs to be built on the same set of nodes
# In a real application, you would ensure the node list is consistent
# Let's get the nodes from our index to ensure consistency
from llama_index.core.node_parser import SentenceSplitter

parser = SentenceSplitter.from_defaults()
nodes = parser.get_nodes_from_documents(documents)

# Create the BM25 retriever from the nodes
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=100)

# Retrieve based on keywords
bm25_nodes = bm25_retriever.retrieve(query)

# In a full hybrid pipeline, you would combine `retrieved_nodes` and `bm25_nodes`
# and then rerank the combined set.
# This ensures both semantic and keyword matches are considered.
# For simplicity here, we'll demonstrate the reranking step next.

print(f"BM25 retrieved {len(bm25_nodes)} nodes based on lexical search.")
```

### 2.3. Analysis and Trade-offs

* **Pros:**

* **Keyword Precision:** Excellent at finding documents with specific, important keywords (e.g., "Form 10-K," "EBITDA," "CEO Name").
* **Complementary to Dense Search:** Catches relevant documents that vector search might miss if the semantic meaning is ambiguous.
* **Computationally Efficient:** Less resource-intensive than deep learning models.

* **Cons:**

* **Lacks Semantic Understanding:** Cannot comprehend synonyms or paraphrasing. A query for "company earnings" will not match a document that only discusses "corporate profits."
* **Vocabulary Mismatch Problem:** Relies on the exact words in the query appearing in the document.

## Section III: Advanced Semantic Reranking with Cross-Encoders

After the initial retrieval stage has recalled a broad set of 100 candidate chunks, and we've potentially fused them with results from a lexical search like BM25, the next critical step is to apply a powerful reranker. This stage shifts the focus from recall to precision, aiming to identify the most relevant chunks from the candidate pool and discard the rest.

### 3.1. Conceptual Framework: Cross-Encoders for Superior Relevance Scoring

While bi-encoders (like BGE) are excellent for efficient first-stage retrieval, they have a limitation: they encode the query and the document independently. **Cross-encoders** overcome this by processing the query and a document *together* in a single pass. This allows the model to perform deep, token-level attention across both texts, resulting in a much more accurate and nuanced relevance score.

For example, a model like `BAAI/bge-reranker-large` is specifically fine-tuned for this task. It takes a (query, document) pair as input and outputs a single score representing the document's relevance to the query. Because cross-encoders are computationally expensive, it would be impractical to run them on the entire corpus. Their ideal use case is as a *reranker* on a smaller, pre-filtered set of candidate documents, such as the 100 chunks retrieved in our first stage.

Similarly, a powerful general-purpose LLM can be used for reranking. Models like Qwen have demonstrated strong reasoning capabilities that can be adapted to this task. LlamaIndex’s `LLMRerank` postprocessor uses the LLM's own logic to reorder documents based on their relevance, often by prompting it with specific instructions.

### 3.2. Strategic Implementation: BGE and Qwen Rerankers

LlamaIndex provides dedicated postprocessors for reranking. The `FlagEmbeddingReranker` is optimized for BGE rerankers, while `LLMRerank` is a more general-purpose solution that can be adapted for models like Qwen.

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

# Using the BGE Reranker
# This model is lightweight and highly effective
bge_reranker = SentenceTransformerRerank(
model="BAAI/bge-reranker-large",
top_n=8  # We want the top 8 most relevant chunks
)

# `retrieved_nodes` are the 100 nodes from the initial vector search
reranked_nodes_bge = bge_reranker.postprocess_nodes(
retrieved_nodes,
query_bundle=llama_index.core.QueryBundle(query_str=query)
)

print("\n--- BGE Reranker Results ---")
for node in reranked_nodes_bge:
print(f"Score: {node.score:.4f}, Text: {node.get_content()[:100]}...")

# For Qwen, we can use the more general LLMRerank if a dedicated
# reranker class isn't available. This uses the LLM's intelligence to score.
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.openai import OpenAI # As a proxy for a Qwen API call

# This requires a powerful LLM. For this example, we'll configure
# it with an OpenAI model, but you would point it to your Qwen endpoint.
# service_context = ServiceContext.from_defaults(llm=QwenLLM(...))
# For demonstration:
Settings.llm = OpenAI(model="gpt-4-turbo")

qwen_style_reranker = LLMRerank(
choice_batch_size=5, # Process 5 chunks at a time
top_n=8,             # Return the top 8 chunks
)

reranked_nodes_qwen = qwen_style_reranker.postprocess_nodes(
retrieved_nodes,
query_bundle=llama_index.core.QueryBundle(query_str=query)
)

print("\n--- Qwen-style LLM Reranker Results ---")
for node in reranked_nodes_qwen:
print(f"Score: {node.score:.4f}, Text: {node.get_content()[:100]}...")
```

### 3.3. Analysis and Trade-offs

| Feature | BGE Reranker (Cross-Encoder) | Qwen Reranker (LLMRerank) |
| :--- | :--- | :--- |
| **Mechanism** | Specialized cross-encoder fine-tuned for relevance scoring. | General-purpose LLM prompted to evaluate and score documents. |
| **Performance** | Extremely fast and accurate for the specific task of reranking. | Potentially more powerful for complex, nuanced queries but slower. |
| **Cost/Latency** | Low. The model is small and optimized. | High. Requires calls to a large, expensive LLM. |
| **Flexibility** | Limited to its pre-trained function of relevance scoring. | Highly flexible. Can be prompted to rerank based on other criteria (e.g., date, style). |
| **Best For** | Production pipelines where speed, cost, and accuracy are balanced. | Scenarios requiring deep reasoning or customizable reranking logic. |

## Section IV: Decomposing Complexity with Prompt Fusion
Complex user queries often contain multiple, distinct questions or constraints. For example, a financial analyst might ask: "<span style="color: #b91c1c; font-weight: bold; background: #fef2f2; padding: 2px 4px; border-radius: 4px;"> Compare the revenue growth and EBITDA margins for TINYCORP and BIGCORP in their latest 10-K filings.</span>
" A single vector search for this entire query might struggle to find documents that address all sub-questions equally well. Prompt fusion, or query decomposition, is a technique that breaks down a complex query into simpler sub-queries, retrieves documents for each one independently, and then fuses the results.

### 4.1. Conceptual Framework: Divide and Conquer

The core idea is to use an LLM to "reason" about the user's query and decompose it. For the example query, the LLM might generate the following sub-queries:

1.  "TINYCORP revenue growth latest 10-K"
2.  "TINYCORP EBITDA margins latest 10-K"
3.  "BIGCORP revenue growth latest 10-K"
4.  "BIGCORP EBITDA margins latest 10-K"

A retriever is then run for each of these sub-queries. The retrieved node sets are combined, duplicates are removed, and the final collection is passed to a reranker to select the best overall context. This approach ensures that all facets of the original complex query are explicitly addressed, significantly improving the thoroughness of the retrieved context.

### 4.2. Strategic Implementation: QueryFusionRetriever

LlamaIndex provides a `QueryFusionRetriever` that automates this entire process. It uses an LLM to generate the sub-queries and then intelligently fuses the results.

```python
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# We will use the vector retriever configured in Section I
vector_retriever = index.as_retriever(similarity_top_k=10)

# The fusion retriever will generate multiple queries and fuse the results.
# It internally uses an LLM to generate the different queries.
fusion_retriever = QueryFusionRetriever(
retrievers=[vector_retriever],
similarity_top_k=10,
num_queries=4,  # Generate 4 sub-queries from the original
mode="reciprocal_rank", # The fusion mode determines how results are combined
use_async=True,
verbose=True,
)

# Use the same reranker from the previous section to refine the fused results
query_engine = RetrieverQueryEngine.from_args(
fusion_retriever,
node_postprocessors=[bge_reranker] # Using the efficient BGE reranker
)

complex_query = "Compare the revenue growth and EBITDA margins for TINYCORP in fiscal year 2023."
response = query_engine.query(complex_query)

print("\n--- Query Fusion Results ---")
# The 'response' object contains the final answer, but we can inspect the source nodes
for node in response.source_nodes:
print(f"Score: {node.score:.4f}, Source: {node.metadata.get('report_type')}")
```

### 4.3. Analysis and Trade-offs

* **Pros:**

* **Comprehensive Recall:** Drastically improves the chances of finding relevant context for all parts of a multi-faceted query.
* **Handles Ambiguity:** By generating multiple phrasings of a query, it can overcome the vocabulary mismatch problem.
* **Synergistic:** Works exceptionally well when combined with a reranker to distill the final context.

* **Cons:**

* **Increased Latency & Cost:** Incurs multiple LLM calls (for query generation) and multiple retrieval operations, making it slower and more expensive than a single query.
* **Query Generation Quality:** The effectiveness of the technique is highly dependent on the LLM's ability to generate meaningful and distinct sub-queries.

## Section V: Optimizing the Input: Query Rewriting and Transformation

Not all parts of a user's prompt are useful for retrieval. Prompts often contain conversational filler, instructions about the output format, or persona-setting phrases ("You are a helpful financial analyst..."). These elements can introduce noise into the embedding space, leading the retrieval system to fetch documents based on these irrelevant instructions rather than the core informational need. Query rewriting is the practice of using an LLM to refine the raw user prompt into an optimal, keyword-rich query targeted specifically for retrieval.

### 5.1. Conceptual Framework: From Conversational to Foundational

The goal is to transform a potentially verbose or poorly phrased user input into one or more concise, focused search queries. For instance, consider the prompt:

> "Hey, can you please act as a world-class financial analyst and quickly summarize the main competitive risks for TINYCORP mentioned in their filings from the last year? Please output the answer as a markdown list. Here's an example of what I want: ..."

A query transformation step would strip away the conversational filler and instructions, isolating the core informational need to produce a clean retrieval query like: "competitive risks for TINYCORP in financial filings from the past year". This refined query is far more likely to produce accurate results when passed to the vector store.

### 5.2. Strategic Implementation: Query Pipelines

LlamaIndex’s `QueryPipeline` provides a flexible, declarative way to chain components together, making it perfect for implementing query transformations. We can create a pipeline that first sends the user query to an LLM with a specific "meta-prompt" for rewriting, and then passes the transformed query to the retriever.

```python
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI

# 1. Define the prompt template for query rewriting
rewrite_template_str = (
"Given a user query, your task is to extract the core informational intent "
"and rephrase it as an optimized, keyword-rich search query for a vector database. "
"Remove all conversational filler, output format instructions, and persona requests. "
"Focus only on the essential concepts, entities, and timeframes.\n"
"Original Query: {query_str}\n"
"Rewritten Query: "
)
rewrite_template = PromptTemplate(rewrite_template_str)

# 2. Define the pipeline components
# For demonstration, we use OpenAI. In practice, this could be any capable LLM.
llm = OpenAI(model="gpt-4-turbo")
p = QueryPipeline(chain=[rewrite_template, llm], verbose=True)

# 3. Define the full RAG pipeline
# This connects the rewriter to the retriever
retriever = index.as_retriever(similarity_top_k=10)

# Build a new pipeline that chains the rewriter and the retriever
full_pipeline = QueryPipeline(
chain=[p, retriever, bge_reranker],
verbose=True
)

# 4. Execute the pipeline with a noisy, conversational query
user_query = (
"Hi there, please act as a senior financial analyst and tell me everything you "
"can find about TINYCORP's primary risk factors from their 2023 annual report. "
"I need the answer in a bulleted list format."
)

final_nodes = full_pipeline.run(query_str=user_query)

print("\n--- Query Rewriting and Retrieval Results ---")
for node in final_nodes:
print(f"Score: {node.score:.4f}, Text: {node.get_content()[:100]}...")
```

### 5.3. Analysis and Trade-offs

* **Pros:**

* **Reduces Noise:** Prevents the retrieval system from being distracted by irrelevant parts of the prompt.
* **Improves Precision:** A focused query leads to more accurate and relevant retrieved chunks.
* **Handles Poor Phrasing:** Can correct user typos or rephrase ambiguous questions into clearer search terms.

* **Cons:**

* **Added Latency/Cost:** Introduces an extra LLM call at the beginning of the process.
* **Risk of Information Loss:** A poorly designed rewrite prompt could potentially strip out a crucial but subtle detail from the user's original query.

## Conclusion and Synthesis: Building a Coherent, Multi-Stage Pipeline

We have explored five distinct but complementary strategies for optimizing the retrieval stage of a financial RAG system. No single technique is a silver bullet; true state-of-the-art performance is achieved by intelligently combining them into a multi-stage pipeline that progressively refines context from a broad search to a highly precise set of evidence.

A robust, production-grade retrieval architecture built with LlamaIndex would look like this:

1.  **Query Transformation:** The raw user query is first passed to a **Query Rewriter** (Section V) to distill its core informational intent, producing a clean, optimized search query.
2.  **Decomposition & Fusion:** The clean query is then fed into a **QueryFusionRetriever** (Section IV), which generates multiple sub-queries to cover all facets of the request.
3.  **Hybrid Retrieval:** For each sub-query, retrieval is performed using a hybrid approach. A **Vector Retriever** (Section I) with metadata filters is run to find semantically similar chunks, while a **BM25 Retriever** (Section II) finds lexically precise matches. The results from both are fused.
4.  **Coarse Reranking:** The combined, deduplicated set of chunks from all sub-queries (which could be several hundred) is passed to an efficient reranker like **BM25** to perform an initial, fast reordering.
5.  **Fine-Grained Reranking:** The top 50-100 chunks from the coarse reranking stage are then passed to a highly accurate **Cross-Encoder Reranker** (Section III) like `bge-reranker-large`. This final step selects the top 8-16 most relevant chunks with high precision.
6.  **Generation:** This final, highly-relevant, and concise set of context is provided to the generator LLM to synthesize the final answer.

The following table provides a summary of the recommended use case for each strategy within the pipeline:

| Strategy | Primary Role | Pipeline Stage | Key Benefit |
| :--- | :--- | :--- | :--- |
| **Vector Search + Metadata** | Initial Recall | 1 (Retrieval) | Broadly captures all potentially relevant semantic information. |
| **BM25** | Lexical Precision | 1 (Retrieval) / 2 (Rerank) | Guarantees matches for critical keywords and technical terms. |
| **BGE/Qwen Reranker** | Precision | 3 (Final Rerank) | Deeply analyzes and scores the top candidates for maximum relevance. |
| **Prompt Fusion** | Comprehensiveness | 0 (Pre-Retrieval) | Ensures all parts of a complex, multi-faceted query are addressed. |
| **Prompt Rewriting** | Noise Reduction | 0 (Pre-Retrieval) | Cleans and focuses the user query for optimal retrieval performance. |

By architecting the retrieval process in this layered manner, data scientists and engineers can build financial RAG systems that are not only powerful and automated but also precise, reliable, and trustworthy—meeting the high standards required in the financial domain.

---

**📚 Continue Your RAG Journey:**  
← **Previous:** [Architecting TinyRAG: A Production-Ready Financial RAG System](/post/architecting-production-ready-financial-rag-system)  
→ **Next:** [A Framework for Rigorous Evaluation and Agentic Post-Processing](/post/framework-rigorous-evaluation-agentic-postprocessing-financial-rag)  
📋 **[View Complete Learning Path](/knowledge/rag)** | **Progress: 7/9 Complete** ✅✅✅✅✅✅✅

