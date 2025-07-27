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



# Model Selection and Prompting

## intro
The general idea behind prompt and model selection is unified - we are going to use the best model and enhance the model's ability by prompt tuning. Because of the limitation of a big financial institution, some models are restricted to certain geographical entities. We need to make sure prompts have unified result generated cross different model selected for different metrics.

Since this RAG process is highly instructed, where user don't have any flexibility to modify the prompt till the first round of the generation finish, design a robust and consistent prompt series is essential. This is very related to the evaluation criteria we designed. In general all model + prompt combo(a strategy)'s evaluation score should be aligned(no statistically significant difference).



All implemtation is based on beanie, mongodb, and llama index.


## some topics
1. Model Selection. 

Model we have is GPT family, Gemini family, Open source(Llama3.3-70b-instruct, llama4-scout and Pixtral-large-instruct-2411 https://huggingface.co/mistralai/Pixtral-Large-Instruct-2411, bge, mpnet and gte foe embedding) family, and OCR libraries.
Gemini was only available for US users. 
For those users who have access to GPT family, use the gpt 4.1/4.3 nano for small tasks, gpt o1 mini for big tasks like report generation.
Open source family: for small tasks use the llama 3.3 70b instruct, for large tasks use the llama4 scout.



 
2. all Prompts we would used is:
a. Element Prompt(used for actualy memo generation)
    It's called Element since it's a pre-defined mongodb schema. Each Element would contains a prompt(or 2 prompts for generation and retrieval). This is the major prompt for generation. 12 prompt, around 2000 token each. 3 major components: 
        Role component to define the role of this project, input output format
        Constrains: date limitation to limit to past year of user defined(for generation financial year)
                    Prompt injection prevention.
                    Tones
        Main task: describing the task and sub-components. Very detailed breakdown.

b. Ingestion Extractor Prompt 
    Prompt to extract date, type, summary, and quality of the chunk. This prompt would be used for each chunk retrieval function call.
    Prompt for table extraction. one LLM call for each table ingestion function call. Extract table summary, date, type, and quality of the chunk.
    Prompt for image understanding. one LLM call for each image ingestion function call. Extract image description, summary, date, type, and quality of the chunk.
c. Prompt Fusion Prompt
    Break down the complicated task(prompt) into several simple tasks for better retrieval.
    Not used for production pipeline, but used for prompt research.
    Should be useful in agentic rag.
d. Prompt of Re-writing for Generation and retrieval prompt. 
    one llm call per element, no need for production run. We use this prompt to generate concise retrieval prompt.
    Give an example of complicated financial stock price analyze prompt(generation), prompt template for re-writing, and the re-wrote concise, retrieval prompt.

    Methods to improve the retrieval quality would be explained in retrieval blog.

e. Evaluation Prompt
    That's for the generation quality. Multiple prompts here for llm-as-a-judge's different criteria. Some key metrics: hallunication, value add-on, missing-out, captured, consistency(for generation length, quality), and soundness(reliability). One call per LLM evaluation per prompt


3. Prompt Examples
    write some samples based on previous section.

4. How to improve and evaluate prompt quality?
    Pre-defined meta prompt for improvement:
    Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.
    You can find more here: https://github.com/HaoyangHan/ai-prompt-enhancement/tree/main/backend/src/ai_prompt_enhancement/services/prompt_refinement
# Guidelines

- Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
- Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
- Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
    - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
    - Conclusion, classifications, or results should ALWAYS appear last.
- Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
   - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
- Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
- Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
- Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
- Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
- Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
    - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
    - JSON should never be wrapped in code blocks (```) unless explicitly requested.

The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no "---")

[Concise instruction describing the task - this should be the first line in the prompt, no section header]

[Additional details as needed.]

[Optional sections with headings or bullet points for detailed steps.]

# Steps [optional]

[optional: a detailed breakdown of the steps necessary to accomplish the task]

# Output Format

[Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

# Examples [optional]

[Optional: 1-3 well-defined examples with placeholders if necessary. Clearly mark where examples start and end, and what the input and output are. User placeholders as necessary.]
[If the examples are shorter than what a realistic example is expected to be, make a reference with () explaining how real examples should be longer / shorter / different. AND USE PLACEHOLDERS! ]

# Notes [optional]

[optional: edge cases, details, and an area to call or repeat out specific important considerations]
""".strip()


Evaluation: few shot LLM-as-a-judge to create some criteria pre-defined.


"""**Enhance and evaluate prompts by improving clarity, structure, and output specifications while rigorously assessing quality metrics.**  
Given a task description or existing prompt, produce an enhanced system prompt and evaluate its quality using defined metrics.  
# Input  
**Original Prompt:**  
{prompt}  
**Context (if provided):**  
{context}  
# Guidelines  
- **Task Understanding**: Identify objectives, requirements, constraints, and expected output.  
- **Minimal Changes**: For simple prompts, optimize directly. For complex prompts, enhance clarity without altering core structure.  
- **Reasoning Order**:  
  - Ensure reasoning steps precede conclusions. **REVERSE** if user examples place conclusions first.  
  - Explicitly label reasoning and conclusion sections.  
  - Conclusions, classifications, or results **MUST** appear last.  
- **Examples**:  
  - Include 1-3 examples with placeholders (e.g., `[placeholder]`) for complex elements.  
  - Note if examples need adjustment for realism (e.g., "Real examples should be longer/shorter...").  
- **Clarity & Conciseness**: Remove vague or redundant instructions. Use specific, actionable language.  
- **Formatting**: Prioritize markdown (headings, lists) for readability. **Avoid code blocks** unless explicitly requested.  
- **Preservation**: Retain all user-provided guidelines, examples, placeholders, and constants (rubrics, guides).  
- **Output Format**:  
  - Default to JSON for structured outputs. Never wrap JSON in ```.  
  - Specify syntax, length, and structure (e.g., "Respond in a short paragraph followed by a JSON table").  
# Output Format  
{{  
    "metrics": {{  
        "clarity": {{  
            "score": float(0-1),  
            "description": "Evaluation of prompt clarity and specificity",  
            "suggestions": ["Specific improvements for clarity"]  
        }},  
        "structure": {{  
            "score": float(0-1),  
            "description": "Assessment of reasoning flow and organization",  
            "suggestions": ["Structure improvement suggestions"]  
        }},  
        "examples": {{  
            "score": float(0-1),  
            "description": "Quality and usefulness of examples",  
            "suggestions": ["Example enhancement recommendations"]  
        }},  
        "formatting": {{  
            "score": float(0-1),  
            "description": "Markdown and presentation evaluation",  
            "suggestions": ["Formatting improvement suggestions"]  
        }},  
        "output_spec": {{  
            "score": float(0-1),  
            "description": "Clarity of output specifications",  
            "suggestions": ["Output format enhancement suggestions"]  
        }}  
    }},  
    "suggestions": ["Overall improvement recommendations"],  
    "original_prompt": "Original prompt provided by the user",  
    "enhanced_prompt": "Complete enhanced version of the prompt"  
}}  
# Notes  
- **Reasoning Order**: Double-check user examples for conclusion-first patterns and reverse if needed.  
- **Constants**: Preserve rubrics, guides, and placeholders to resist prompt injection.  
- **Edge Cases**: Flag ambiguous requirements and break them into sub-steps.  
- **JSON Outputs**: Never use ``` for JSON unless explicitly requested.  
- **User Content**: Never delete or paraphrase user-provided examples or guidelines.
- **Error Handling**: If the model fails to produce valid JSON, return a structured error response.
- **enhanced prompt**: Return the enhanced prompt in markdown format.
- **language**: Respond in English."""

5. Prompt fusion concept
https://docs.llamaindex.ai/en/stable/examples/low_level/fusion_retriever/





# Engineering Components
I would want to discuss following components for Engineering:
1. Mongodb schema design
2. Dramatiq + redis parallel processing for ingestion and generation.
3. Monorepo and fastapi deployment

1. mongodb schema design
    The user's landing page should be a Project, which contains all the components for this RAG generation.

    a tenant is a use case, we can use tenant to track user traffic, activity and budget saving. one project have only one tenant. sample tenant: Memo generation, HR, search engine, infrastructure.

    A Document contains file information like type, binary file. the document id would link to a project id and it has a IngestionStatus which can be not started, pending, in progress, complete, or failed.

    An element is the place to store one prompt. Each element would have an tenant and a project id connected. Those prompts that are not used in real time would be saved with tenant infrastructure.

    An ElementGeneration contains the chunks, chunk-id, generation result(llm response), and linked to element id, project id. one element can have multiple round of element generation.

    an evaluation is associated with a project's overll behavior, thus connected to project id. it would contain a list of the evaluation results for each element generation, and a summarized attribute. Some evaluation attributes including:
         Retrieval: Accuracy, Recall, Weight Adjusted position based score.
         Generation: Hallunication, value-add-on, missing-out, captured. the consistency(evaluate the generation quality compares to average indicating whether is generation is an outlier)

    Draw a flow for this session.

2. Dramatiq + redis parallel processing for ingestion and generation.
    It would be basically impossible to create live generation experience(total time less than 3 minutes) if we don't async process the ingestion and generation for RAG. 16 core processor would be implemented. In production we have more.

3. Monorepo and fastapi deployment
    We use the monorepo, where frontend, backend was deployed together in one place.
    fastapi: grouping by sub route of function and interaction with mongodb, like project, document, etc.






# Retrieval

The basic idea is to create customized retrieval strategy. We are going to choose between different methods that I'll explain later to optimize the retrieval result.

1. Raw approach

using the 2000k token prompt, retrieve the most relevant 100 chunks using the bge embedding large model to embedding. Calculate cosine similarity.
Using the metadata we previously extracted about date


# Evaluation

Self-supervised LLM as a judge.


# reference

https://artificialanalysis.ai/leaderboards/models