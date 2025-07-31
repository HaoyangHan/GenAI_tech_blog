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
All based on llama index, python

1. Raw approach

using the 2000k token prompt, retrieve the most relevant 100 chunks using the bge embedding large model to embedding. Calculate cosine similarity.
Using the metadata we previously extracted about date, type, keywords. Use those as extra score increase/decrease method.
Reranking and get the top 8 or 16 chunks for generation.

2. bm25 reranker
using bm 25 reranker after 100 chunk retrieval, explain the benifit of using it, return 8/16 chunks
give llama index implementation

3. bge or qwen3 reranker.

4. prompt fusion - decompose the complex prompt to some simple prompts, retrieval for each part, then using rerank model to combine and select final chunks

5. prompt re-write
change the complicated and unrelated part(like: you are acting as a financial analyst.....; just focus on the data for past year; output is markdown; here are some examples) should be cleaned. 
use a meta prompt to re-write the generation prompt to retrieval prompt(easy, straightforward)


# Evaluation and post processing

Evaluation is basically 

For retrieval evaluation, basically search or classification problem where we evaluate whether the chunks that contains important message correctly from retrival process.

evaluate the original retrieval: human expert label 8 chunks that are most important by hand, we would see whether those chunks are in the 100 candidates. percision, recall, F1 other metrics if possible
evaluate the reranking retrieval: similarly, but the candidate is 8/16 instead of 100. Also, if the chunks are not in the original candidates, they should not be count in reranking metrics. generate same metric.

Evaluate generation
2 phase: 
human supervised data collection(where financial expert would use the historical year memo as evaluation data). 50+(projected from 5% of 1000 projects we would use annually in the session here: posts/Business Objective/project_mvp_workflow.md) memos are evaluated.
afterwards, we learned human expert pattern on how to evaluate, then created few-shot llm-as-a-judge auto evaluation methods to collect more data. basically, learn the pattern, provide llm examples on what is  the 5, 3, 1 score critera for each evaluation critera.
Criteria: 
1. hallunication: whether model made a number, fact that does not exist in chunk.
2. captured rate: the real important messages that are captured both by human and RAG system.
3. value add: the important fact RAG captured but human didn't
4. missing out: the important fact human captured but RAG didn't
5. confidence level: detect the confidence level of each part of the generation. if it's low or medium, give human a hint. this is the extension of hallunication.

Evaluate Consistency:

we generated a set of scores based on the model selection. for one memo generation, different model should have aligned performance cross multiple metrics. we have gemini, openai, open source solution, and their metrics should be aligned.

* **OpenAI (GPT & o-series):** This family represents the established benchmark for high-quality, general-purpose language and reasoning models. The `GPT-4.1` series (`nano`, `mini`, and the full model) offers significant improvements over its predecessors in coding, instruction following, and long-context comprehension, all while supporting a 1 million token context window.[6, 7] The `o-series`, particularly `o1-mini`, is specialized for complex reasoning tasks, excelling in STEM and programming challenges where it allocates more computational "thought" to derive solutions.[8, 9] This makes the OpenAI family a reliable, high-quality baseline for a wide range of tasks, from high-throughput classification to deep analytical report generation. Their global availability and mature API ecosystem further solidify their role as a foundational component of a multi-provider strategy.[10, 11]

  * **Google (Gemini):** The Gemini family, particularly models like `Gemini 2.5 Pro`, is distinguished by its frontier-level multimodal capabilities and an exceptionally large context window, capable of processing entire books or extensive codebases in a single pass.[12] This makes it uniquely suited for tasks involving the deep analysis of mixed-media financial reports, video earnings calls, or complex visual data. However, its utility within a global financial institution is critically hampered by significant geographical availability restrictions. As of early 2025, access to certain Gemini models and features via Vertex AI remains limited, with a primary focus on the North American market and potential deprecation for new projects in other regions.[13, 14]

  * **Open-Source (Llama, Pixtral):** The open-source ecosystem, led by models from Meta and Mistral AI, represents a strategic imperative for any organization seeking to mitigate vendor lock-in, enhance data privacy, and gain granular control over its AI stack. These models are not merely "free" alternatives but are increasingly competitive and architecturally innovative. Meta's `Llama 4 Scout` employs a Mixture-of-Experts (MoE) architecture, which allows for highly efficient inference on a very large model by activating only a subset of its parameters for any given task.[15, 16] Mistral AI's `Pixtral-Large-Instruct-2411` is a natively multimodal model demonstrating state-of-the-art performance on document understanding benchmarks like `DocVQA`.[17, 18, 19] Deploying these models on internal infrastructure provides complete control over data flow and allows for deep customization and fine-tuning on proprietary financial data, offering a powerful complement to commercial APIs.


Creat another session for agentic post processing.
llama index as example.
Give example after we have llm-as-a-judge-score, how to do agentic post processing.
Bacially everything in ingestion, retrieval, and generation is the traditional rag. After we get the evaluation, LLM would be able to determine state to do based on score.
If score is low for one element generation(we have 12 sections, which is 12 components, or segmentations of memo generation), we can use different method to diagnosis and improve the generation(do regeneration). for example, we can use the prompt fusion, or re-writing, or expand the chunk reranking number, or use more powerful model like openai's o3. for re-generation and re-evaluation.
If the score is medium, we can involve human in the loop, to let our slack or discord bot send message to human indicating whether this memo is good or not.
If the score is high, we can do post processing, like aggregate the segmentations into full report in docx and pdf, send via email, convert to slides and html.



# reference

https://artificialanalysis.ai/leaderboards/models