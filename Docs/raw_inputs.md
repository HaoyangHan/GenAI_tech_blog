Ingestion

ingestion document source
we talked about the input of this project where input is 
This includes structured data from spreadsheets (financial conditions), semi-structured data from credit reports, and vast unstructured text from SEC filings and historical memos. in detail they could be:
text pdf. this is around 30 - 60 page on average(25% to 75% percentile), each page is around 350-500 words, that's 700 - 1000 tokens.
use the pymupdf to load the document object. https://github.com/pymupdf/PyMuPDF

Chunk: chunk is the concept of a combination of text + embedding word vectors. Chunking strategy: 1024 tokens + 10% overlap. Advanced: semantic chunking: https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/

Metadata extraction:
we would extract the keywords, date, and, summary, chunk quality(based on the text corpus quality) out for each chunk using one LLM call. return json only. design this prompt.

Special format:
tables are saved as a special chunk
Table(csv) save to markdown format using one LLM call.
Table(in pdf, text format): recognize using pymupdf when we split the chunk. create a special prompt for all tables to write understanding of the numbers in this table, including trends.
Image, image based pdf: a special chunk. baseline handling method is OCR. If image paged pdf, each chunk is one image(page). Advanced mehtod: using gpt4o, gemini-2.5-pro, or llama-4-scout do image to text understanding.
pptx(slides): 


embedding model

bge large: https://huggingface.co/BAAI/bge-large-en
bge multilingual: https://huggingface.co/BAAI/bge-multilingual-gemma2
bge reranker: https://huggingface.co/BAAI/bge-reranker-large
qwen3 embedding model and reranker model.


Storage:

Chunks: embedding concept and where to store the embedding is chromadb: https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo/
Documents: saved under mongodb using beanie to connect: from datetime import datetime
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
    """A chunk of text from a document."""
    text: str
    page_number: int
    chunk_index: int
    embedding: Optional[List[float]] = None

class Document(Document):
    """Document model for storing uploaded documents and their chunks."""
    user_id: Indexed(str)
    project_id: Indexed(str)
    
    # Required fields for MongoDB validation
    filename: str
    content_type: str
    file_size: int
    status: str = "processing"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Additional fields
    metadata: DocumentMetadata
    chunks: List[DocumentChunk] = []
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_deleted: bool = False

    class Settings:
        name = "documents"
        indexes = [
            "user_id",
            "project_id",
            "filename",
            "status",
            "created_at"
        ]

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "project_id": "project123",
                "filename": "example.pdf",
                "content_type": "application/pdf", 
                "file_size": 1024,
                "status": "completed",
                "metadata": {
                    "filename": "example.pdf",
                    "content_type": "application/pdf",
                    "size": 1024,
                    "upload_date": "2024-02-20T12:00:00",
                    "processed": True
                },
                "chunks": [
                    {
                        "text": "Example document chunk",
                        "page_number": 1,
                        "chunk_index": 0
                    }
                ]
            }
        } 


since this blog is not completed yet, write a summary session as memory so we can continue.


Special things to discuss:

Topic 1. Why 1024 token size per chunk in the begining?

reverse engineering. In 2024, lot's of the smaller model has content size of 16k input tokens and a sliding window(if the total conversation is larger than 16k token, it would only use the latest 16k). we would generate around 400 - 1200 tokens for each segmentaion, 2k token prompt(yes we have very complicated prompt), 8 chunks in the retrieval, so we picked 1024 as the initial design.
Later on we used some other methods of semantic chunking and page based chunking, but it's the further improvement.
List some openai, llama3.x, gemini model that is 16k tokens input


Topic 2. What exact number of document we have and how many chunks we have? estimated time eclapsed?
3-10 documents, 20 document max. 
several pages to 200 pages. on average 30 page per document
1 minute or so average document processing time.
Majority English(by default we only support english unless specified and using bge-multilingual).

Topic 3. impact about using chromadb versus using postgresdb
20 % time slower for entire processing(document upload, metadata extraction, image processing, embedding, save to database)


topic 4. What If Date and Type not extracted?
Force user to select. if they didn't select, we would not add or decrease value to this chunk.


topic 5. Engineering design of async processing the chunk
The tech stack is async define a background task, using redis to handle the multi-processing, and using dramatiq to host the multiple tasks.


# Prompting
Prompts we would used is:
Prompt to extract date, type, summary, and quality of the chunk. This prompt would be used for each 

# Engineering Components

# Retrieval

The basic idea is to create customized retrieval strategy. We are going to choose between different methods that I'll explain later