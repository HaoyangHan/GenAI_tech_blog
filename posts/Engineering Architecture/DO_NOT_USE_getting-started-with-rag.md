---
title: "Architecting TinyRAG: A Production-Ready Financial RAG System"
category: "Engineering Architecture"
date: "July 27, 2025"
summary: "This post details the production architecture for a financial RAG system, moving beyond prototypes. We architect a robust, decoupled system using a web-queue-worker pattern with FastAPI, Dramatiq, and MongoDB to handle high-throughput ingestion and reliable querying, underpinned by a rigorous evaluation framework."
slug: "architecting-production-ready-financial-rag-system"
tags: ["RAG", "Python", "LlamaIndex", "FastAPI", "MongoDB", "Dramatiq", "Redis", "Financial AI", "System Design"]
author: "Haoyang Han"
---

## Introduction: Beyond the Prototype

This post details the architecture and implementation of **TinyRAG**, our robust, scalable financial RAG system designed to meet these stringent requirements. The architectural thesis is built on a foundational principle: <span style="color: #34A853;">***decoupling for scale and reliability***</span>. We will employ a **web-queue-worker** architecture, a pattern proven in countless production systems. This design separates the user-facing API from the intensive background processing. A lightweight FastAPI web service handles incoming requests, while computationally expensive tasks—document parsing, embedding, and indexing—are offloaded to a resilient, horizontally scalable backend powered by Dramatiq and Redis.

## The 'Why': Data Science Thinking & Architectural Decisions

Before diving into code, it's critical to understand the strategic decisions that shape the system. A prototype might bundle everything into a single script, but a production system must be designed for resilience, scalability, and maintainability.

### The Architectural Blueprint: A Decoupled System

Our choice of a **web-queue-worker** architecture is deliberate. The primary challenge in production RAG is handling asynchronous, long-running tasks (document ingestion) without compromising the responsiveness of the synchronous, user-facing tasks (querying).

| Component | Role in TinyRAG | Justification & Trade-offs |
| :--- | :--- | :--- |
| **FastAPI Web Service** | User-facing API Layer | Handles `/ingest` and `/query` requests. Its asynchronous nature allows it to efficiently handle many concurrent connections. By offloading heavy work, it remains lightweight and highly available. |
| **Redis Broker** | Task Queue | Acts as the message broker for Dramatiq. It provides a durable, high-speed buffer for ingestion tasks, ensuring no document is lost even during traffic spikes. |
| **Dramatiq Workers** | Background Processing | The "workhorses" that execute the `LlamaIndex` ingestion pipeline. They can be scaled horizontally and independently of the web API to match the ingestion load, providing architectural elasticity. |
| **MongoDB** | Document & Vector Store | Serves a dual purpose: storing raw document text and rich metadata, and as a scalable vector store for `LlamaIndex` using its Atlas Vector Search capabilities. This simplifies the tech stack. |
| **LlamaIndex** | Core RAG Framework | Provides the high-level abstractions for our `IngestionPipeline` and `QueryEngine`, dramatically accelerating development while offering deep customization options for parsing, embedding, and retrieval. |

This decoupled design ensures that a flood of document submissions for ingestion will not slow down or crash the query service.

### System Flow Diagram

The diagram below illustrates the flow of data for both ingestion and querying, making the separation of concerns clear.

![Engineering Workflow Diagram](/images/engineering/engineering_workflow.png)

*Figure: High-level engineering workflow for TinyRAG's decoupled, production-ready architecture. The diagram illustrates the separation of concerns between the FastAPI web service, Redis-backed Dramatiq workers, and MongoDB, ensuring scalable ingestion and reliable querying.*


### The Data Backbone: Pydantic & The Monorepo Data Contract

A robust system is built on a solid data foundation. For financial documents, _metadata is a first-class citizen_. Storing rich context—source document, page number, report type, ticker—is non-negotiable for providing verifiable answers.

To enforce data consistency across our distributed services (API and workers), we use Pydantic models as a **"single source of truth."** This concept is amplified by our choice of a **Monorepo structure**. Both the FastAPI app and the Dramatiq workers import the same `models.py` file from a shared library. This completely eliminates a common class of integration bugs caused by version skew between services, creating a self-validating and maintainable data flow.

## Implementation: Code & Explanation

Here we translate the architecture into production-quality, asynchronous Python code.

### 1. The Shared Data Contract (`models.py`)

These Pydantic models define our core data structures. We define a custom `PyObjectId` to ensure seamless conversion between Pydantic and MongoDB's `ObjectId`. According to the official Pydantic documentation, this is a standard pattern for handling third-party types.

```python
# app/shared/models.py
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from bson import ObjectId
from pydantic import BaseModel, Field, ConfigDict

# Custom type for handling MongoDB's ObjectId
class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: Any) -> Any:
        from pydantic_core import core_schema
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(ObjectId),
                    core_schema.chain_schema(
                        [
                            core_schema.str_schema(),
                            core_schema.no_info_plain_validator_function(cls.validate),
                        ]
                    ),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x)
            ),
        )

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

class DocumentMetadata(BaseModel):
    """Schema for financial document metadata."""
    report_type: Optional[str] = None # e.g., "10-K", "10-Q"
    ticker: Optional[str] = None
    year: Optional[int] = None
    source_url: Optional[str] = None

class DocumentSchema(BaseModel):
    """Schema for a document entry in MongoDB."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    file_name: str
    status: str = Field(default="pending", description="pending, processing, completed, failed")
    metadata: DocumentMetadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )
```

### 2. High-Throughput Ingestion (Dramatiq & LlamaIndex)

First, we configure the Dramatiq broker to use Redis.

```python
# app/worker/broker.py
import dramatiq
from dramatiq.brokers.redis import RedisBroker
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_broker = RedisBroker(url=REDIS_URL)
dramatiq.set_broker(redis_broker)
```

The FastAPI endpoint is designed to be extremely fast. It creates a document record and immediately enqueues the processing task using `process_document_ingestion.send()`, returning a `202 Accepted` response. This non-blocking behavior is critical for API performance.

```python
# app/api/main.py (Ingestion Endpoint)
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from app.shared.models import DocumentSchema, DocumentMetadata
from app.worker.tasks import process_document_ingestion
from motor.motor_asyncio import AsyncIOMotorClient
import os

app = FastAPI(title="TinyRAG API")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URL)
db = client.financial_rag_db

@app.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(file: UploadFile = File(...)):
    # In a real system, you would save the file to persistent storage like S3
    file_content = await file.read() # Async file read

    doc_metadata = DocumentMetadata(ticker="XYZ", report_type="10-K", year=2023)
    document = DocumentSchema(file_name=file.filename, metadata=doc_metadata)

    # Insert into DB to get the ID
    result = await db.documents.insert_one(document.model_dump(by_alias=True, exclude=["id"]))
    document_id = result.inserted_id

    # Enqueue the task for the Dramatiq worker
    process_document_ingestion.send(str(document_id))

    return {"message": "Document ingestion started.", "document_id": str(document_id)}
```

The Dramatiq actor is where the heavy lifting happens. It fetches the document details and runs the `LlamaIndex` `IngestionPipeline`. Note the use of `await pipeline.arun()` and `num_workers=4`. This provides a <span style="color: #4285F4;">**two-tiered scalability model**</span>: Dramatiq scales *across documents*, while LlamaIndex's `num_workers` scales *within a single large document* by parallelizing chunk processing.

```python
# app/worker/tasks.py
import dramatiq
import os
from motor.motor_asyncio import AsyncIOMotorClient
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from .broker import redis_broker # Ensures broker is configured
from app.shared.models import PyObjectId

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = "financial_rag_db"
VECTOR_COLLECTION_NAME = "chunks"

mongo_client = AsyncIOMotorClient(MONGO_URL) # Each worker process needs its own client

@dramatiq.actor(max_retries=3, time_limit=300_000) # 5 min timeout
async def process_document_ingestion(document_id_str: str):
    document_id = PyObjectId(document_id_str)
    db = mongo_client[DB_NAME]
    
    await db.documents.update_one({"_id": document_id}, {"$set": {"status": "processing"}})
    
    try:
        doc_data = await db.documents.find_one({"_id": document_id})
        # Assume file is in a local 'data' dir for this example
        reader = SimpleDirectoryReader(input_files=[f"data/{doc_data['file_name']}"])
        docs = await reader.aload_data()

        vector_store = MongoDBAtlasVectorSearch(
            client=mongo_client,
            db_name=DB_NAME,
            collection_name=VECTOR_COLLECTION_NAME,
        )
        
        pipeline = IngestionPipeline(
            transformations=[OpenAIEmbedding()],
            vector_store=vector_store
        )

        await pipeline.arun(documents=docs, num_workers=4)

        await db.documents.update_one({"_id": document_id}, {"$set": {"status": "completed"}})
    except Exception as e:
        await db.documents.update_one({"_id": document_id}, {"$set": {"status": "failed", "error": str(e)}})
        raise # Re-raise for Dramatiq retry logic
```

### 3. Serving Queries Asynchronously

The query endpoint loads the `VectorStoreIndex` from our existing MongoDB store. This is a lightweight operation. We use `query_engine.aquery()` and FastAPI's `StreamingResponse` to stream tokens back to the client as they are generated, providing a much better user experience than waiting for the full response.

```python
# app/api/main.py (Query Endpoint)
#... (add these imports)
from fastapi.responses import StreamingResponse
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

# Configure LlamaIndex global settings
Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
Settings.embed_model = OpenAIEmbedding()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def handle_query(request: QueryRequest):
    vector_store = MongoDBAtlasVectorSearch(client=client, db_name=DB_NAME, collection_name=VECTOR_COLLECTION_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = index.as_query_engine(streaming=True)

    streaming_response = await query_engine.aquery(request.query)

    async def event_generator():
        for token in streaming_response.response_gen:
            yield token

    return StreamingResponse(event_generator(), media_type="text/plain")
```

### 4. Implementation: Quantitative Evaluation with Ragas

For a financial system, *hope is not a strategy*. We must quantitatively measure performance. The open-source `ragas` library provides metrics to evaluate the two pillars of RAG: **Retrieval** and **Generation**.

| Metric Category | Metric Name | Why it Matters for TinyRAG |
| :--- | :--- | :--- |
| Retrieval | `Context Precision` | Prevents the LLM from being distracted by irrelevant financial data or boilerplate text. |
| Retrieval | `Context Recall` | Ensures the answer is comprehensive and doesn't omit critical data like a specific risk factor. |
| Generation | `Faithfulness` | **The most critical metric.** Directly penalizes hallucination of financial figures, dates, or events. |
| Generation | `Answer Relevancy` | Ensures the system directly answers the user's specific financial question. |

This evaluation script provides a framework for creating a "golden dataset" and scoring our system's performance. The resulting scores transform system tuning from guesswork into data-driven engineering.

```python
# evaluate.py
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

# In a real project, you'd populate 'contexts' and 'answer' by running your RAG system.
# 'ground_truth' is the manually curated, ideal answer.
def create_evaluation_dataset():
    data = {
        "question": ["What were the key risks mentioned for XYZ in 2023?"],
        "contexts": [["Risk factor A...", "Risk factor B...", "Irrelevant section..."]],
        "answer": ["The key risks were A and B."],
        "ground_truth": ["The key risks for XYZ in 2023 were risk factor A and risk factor B."]
    }
    return Dataset.from_dict(data)

def run_evaluation():
    eval_dataset = create_evaluation_dataset()
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    
    print("Running RAG evaluation...")
    result = evaluate(dataset=eval_dataset, metrics=metrics)
    print("Evaluation Results:")
    print(result) # e.g., {'faithfulness': 0.95, ...}
    return result

if __name__ == "__main__":
    run_evaluation()
```

## Conclusion & Next Steps

We have architected **TinyRAG**, a production-grade financial RAG system that prioritizes scale, reliability, and trustworthiness. By using a <span style="color: #34A853;">decoupled web-queue-worker pattern</span>, enforcing a <span style="color: #34A853;">strong data contract</span>, and building a <span style="color: #34A853;">quantitative evaluation framework</span>, we have laid a foundation that moves far beyond a simple prototype.

This is just the beginning. The next post in this series will dive deeper into one of the most critical parts of the pipeline: **Advanced Retrieval Strategies**. We will explore how to implement and evaluate a `CohereReRank` step to dramatically improve `Context Precision`, ensuring our generation model receives only the most relevant, highest-quality information to craft its answers.

---
### Sourcing and Further Reading
<div id="refs"></div>

1.  Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. arXiv:2005.11401.
2.  LlamaIndex Documentation. (2024). *High-Level Concepts*. [https://docs.llamaindex.ai/en/stable/getting_started/high_level_concepts.html](https://docs.llamaindex.ai/en/stable/getting_started/high_level_concepts.html)
3.  U.S. Securities and Exchange Commission. (2023). *SEC Charges Two Investment Advisers for Making False and Misleading Statements About Their Use of Artificial Intelligence*. [https://www.sec.gov/news/press-release/2024-30](https://www.sec.gov/news/press-release/2024-30)
4.  TruEra. (2023). *The Urgent Need for LLM Evaluation*. [https://truera.com/the-urgent-need-for-llm-evaluation/](https://truera.com/the-urgent-need-for-llm-evaluation/)
5.  Adam Johnson. (2023). *The Web-Queue-Worker Architecture*. [https://adamj.eu/tech/2023/10/18/the-web-queue-worker-architecture/](https://adamj.eu/tech/2023/10/18/the-web-queue-worker-architecture/)
6.  FastAPI Documentation. (2024). *Concurrency and async / await*. [https://fastapi.tiangolo.com/async/](https://fastapi.tiangolo.com/async/)
7.  Dramatiq Documentation. (2024). *User Guide*. [https://dramatiq.io/](https://dramatiq.io/)
8.  Redis Documentation. (2024). *Redis as a message broker*. [https://redis.io/docs/manual/pubsub/](https://redis.io/docs/manual/pubsub/)
9.  LlamaIndex Documentation. (2024). *Ingestion Pipeline*. [https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/root.html](https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/root.html)
10. LlamaIndex Documentation. (2024). *MongoDBAtlasVectorSearch*. [https://docs.llamaindex.ai/en/stable/api_reference/vector_stores/mongodb.html](https://docs.llamaindex.ai/en/stable/api_reference/vector_stores/mongodb.html)
11. Pydantic Documentation. (2024). *Pydantic V2*. [https://docs.pydantic.dev/latest/](https://docs.pydantic.dev/latest/)
12. Ragas Documentation. (2024). *Ragas Metrics*. [https://docs.ragas.io/en/latest/concepts/metrics/index.html](https://docs.ragas.io/en/latest/concepts/metrics/index.html)
13. Esau, S., et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217.
```