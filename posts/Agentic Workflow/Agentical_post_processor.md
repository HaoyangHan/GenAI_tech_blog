---
title: "Agentic Post-Processing in TinyRAG: Architecting Conditional, State-Driven Workflows"
category: "Agentic Workflow"
date: "July 27, 2025"
summary: "This article explores how to move beyond static RAG pipelines by leveraging LlamaIndex's Workflow abstraction to build agentic, event-driven post-processing for robust, production-grade document generation. We detail the design and implementation of a conditional, stateful workflow that autonomously evaluates, regenerates, and escalates outputs based on quality metrics."
slug: "agentic-post-processing-tinyrag"
tags: ["RAG", "Agentic Workflow", "LlamaIndex", "Python", "State Machine", "Post-Processing", "Evaluation", "Production AI", "System Design"]
author: "Haoyang Han"
---

**📚 RAG Implementation Series - Article 9 of 9:**  
[Complete Learning Path](/knowledge/rag) | ← [Previous: Evaluation Framework](/post/framework-rigorous-evaluation-agentic-postprocessing-financial-rag) | **Current: Agentic Workflows** → **Series Complete!** 🎉


(Continue from evaluation)

# Part II: From Evaluation to Action: Agentic Post-Processing with LlamaIndex

## Section 4: Architecting a Conditional, State-Driven Workflow

A static, linear pipeline is insufficient for a production system that demands reliability. The TinyRAG system requires a dynamic, intelligent post-processing layer that can react to the quality of its own outputs. This is the domain of **Agentic RAG**, a paradigm where the system moves beyond a simple "retrieve-then-generate" flow to one that can reason about its performance and take autonomous, corrective actions.³⁰

The logic required—if the evaluation score for a memo segment is low, then regenerate it; if the score is medium, then escalate to a human—is fundamentally a conditional, state-driven workflow. The most robust and elegant way to implement this within the LlamaIndex ecosystem is to use the purpose-built `Workflow` class. This abstraction allows for the definition of a formal, event-driven state machine that serves as the central nervous system for the entire post-processing pipeline.³² This approach is cleaner, more debuggable, and far more extensible than a series of ad-hoc `if/else` scripts.

### 4.1 Designing the Workflow Events and State

The workflow is driven by a series of custom events, which are defined as Pydantic models. This provides type safety, validation, and a clear structure for passing state between different steps of the workflow.³²

The key events for the `MemoProcessingWorkflow` are:

  * `StartMemoGeneration(Event)`: This event initiates the workflow, containing the initial user query and the list of retrieved context chunks.
  * `SegmentGenerated(Event)`: This event is emitted after the LLM generates a single segment of the memo. It contains the generated text and a reference to the original query.
  * `EvaluationComplete(Event)`: This is the pivotal event in the workflow. It is emitted after the LLM-as-a-Judge has scored a segment and contains the generated text along with the full, structured JSON object of evaluation scores.
  * `RegenerationRequired(Event)`: Triggered by the workflow logic when an `EvaluationComplete` event has a low score. It contains the data needed for the regeneration attempt.
  * `HumanReviewRequired(Event)`: Triggered for medium-scored segments. It contains the text, scores, and context needed to format a clear message for human review.
  * `SegmentApproved(Event)`: This event signifies that a segment has met the quality bar, either by achieving a high score initially or by being manually approved by a human expert.
  * `FinalReportReady(Event)`: This event is triggered by a final aggregation step once all required memo segments have been approved. It contains the complete, ordered content for the final memo.
  * `StopEvent(Event)`: A built-in LlamaIndex event that gracefully terminates the workflow, returning the final output (e.g., the paths to the generated document files).

### 4.2 The Core Workflow Logic Skeleton

The entire post-processing logic is encapsulated within a `MemoProcessingWorkflow` class that inherits from `llama_index.core.workflow.Workflow`. The behavior of the workflow is defined by asynchronous methods decorated with `@step`, which specify which events they listen for and which events they can emit.

The following Python code provides the high-level skeleton for this workflow, illustrating how the different steps connect via events.

```python
from llama_index.core.workflow import (
    Workflow,
    Event,
    StartEvent,
    StopEvent,
    step,
    Context
)
from typing import List, Dict, Any

# --- Define Custom Events ---
class SegmentGenerated(Event):
    segment_text: str
    segment_id: str
    context_chunks: List[str]

class EvaluationComplete(Event):
    segment_text: str
    segment_id: str
    evaluation_scores: Dict[str, Any]

class RegenerationRequired(Event):
    segment_id: str
    original_text: str
    evaluation_scores: Dict[str, Any]
    context_chunks: List[str]

class HumanReviewRequired(Event):
    segment_id: str
    segment_text: str
    evaluation_scores: Dict[str, Any]
    context_chunks: List[str]

class SegmentApproved(Event):
    segment_id: str
    approved_text: str

class FinalReportReady(Event):
    all_segments: List[str]

# --- Define the Main Workflow Class ---
class MemoProcessingWorkflow(Workflow):
    @step
    async def generate_and_evaluate(self, ev: StartEvent) -> EvaluationComplete:
        # This step would take the initial query from StartEvent,
        # call the generator LLM to produce a segment, and then
        # call the LLM-as-a-Judge.
        print(f"Step 1: Generating and Evaluating Segment...")
        #... logic to call generator and judge...
        generated_text = "This is the AI-generated memo segment."
        context = ["context chunk 1...", "context chunk 2..."]
        human_ref = "This is the human reference."
        
        # Mocking the judge call from the previous section
        scores = get_llm_judge_evaluation(context, generated_text, human_ref)
        
        return EvaluationComplete(
            segment_text=generated_text,
            segment_id="section_1_intro",
            evaluation_scores=scores
        )

    @step
    def route_based_on_score(self, ev: EvaluationComplete) -> Event:
        # This is the core conditional logic step.
        print(f"Step 2: Routing based on evaluation score...")
        
        # Calculate a composite score for routing decision
        # A simple average for demonstration purposes
        scores = ev.evaluation_scores
        composite_score = sum(s['score'] for s in scores.values()) / len(scores)
        print(f"Composite Score: {composite_score:.2f}")

        if composite_score >= 4.5:
            print("Outcome: High Score. Approving segment.")
            return SegmentApproved(
                segment_id=ev.segment_id, 
                approved_text=ev.segment_text
            )
        elif 2.5 <= composite_score < 4.5:
            print("Outcome: Medium Score. Sending for human review.")
            return HumanReviewRequired(
                segment_id=ev.segment_id,
                segment_text=ev.segment_text,
                evaluation_scores=ev.evaluation_scores,
                context_chunks=["context chunk 1...", "context chunk 2..."] # Pass context for review
            )
        else: # composite_score < 2.5
            print("Outcome: Low Score. Requiring regeneration.")
            return RegenerationRequired(
                segment_id=ev.segment_id,
                original_text=ev.segment_text,
                evaluation_scores=ev.evaluation_scores,
                context_chunks=["context chunk 1...", "context chunk 2..."]
            )
            
    @step
    async def handle_regeneration(self, ev: RegenerationRequired) -> EvaluationComplete:
        # This step handles the low-score path.
        print(f"Step 3a: Handling Regeneration for segment {ev.segment_id}...")
        #... logic to call a RegenerationTool...
        # The tool would return a new, improved text.
        regenerated_text = "This is the new, improved memo segment after regeneration."
        
        # After regeneration, the new text must be re-evaluated.
        print("Re-evaluating regenerated text...")
        context = ev.context_chunks
        human_ref = "This is the human reference."
        new_scores = get_llm_judge_evaluation(context, regenerated_text, human_ref)
        
        # Emit a new EvaluationComplete event to re-enter the routing logic.
        return EvaluationComplete(
            segment_text=regenerated_text,
            segment_id=ev.segment_id,
            evaluation_scores=new_scores
        )

    @step
    async def handle_human_review(self, ev: HumanReviewRequired) -> SegmentApproved:
        # This step handles the medium-score path.
        print(f"Step 3b: Handling Human Review for segment {ev.segment_id}...")
        #... logic to call a SlackNotificationTool and wait for a response...
        # The tool would return the human-approved text.
        print("...Message sent to Slack. Waiting for human approval...")
        # In a real system, this would involve a webhook or polling.
        # For demonstration, we simulate immediate approval.
        approved_text_from_human = ev.segment_text + " [Human Approved]"
        print("Human has approved the segment.")
        
        return SegmentApproved(
            segment_id=ev.segment_id,
            approved_text=approved_text_from_human
        )

    @step
    def aggregate_and_finalize(self, ctx: Context, ev: SegmentApproved) -> Event:
        # This step collects all approved segments.
        print(f"Step 4: Aggregating approved segment {ev.segment_id}...")
        
        # Use the workflow context to store state
        if 'approved_segments' not in ctx.store:
            ctx.store['approved_segments'] = {}
        
        ctx.store['approved_segments'][ev.segment_id] = ev.approved_text
        
        # Assume for this example we only need 1 segment.
        # In a real system, this would check if all expected segments are present.
        TOTAL_SEGMENTS_NEEDED = 1
        if len(ctx.store['approved_segments']) == TOTAL_SEGMENTS_NEEDED:
            print("All segments approved. Preparing final report.")
            # Assume segments are ordered by ID
            final_content = list(ctx.store['approved_segments'].values())
            return FinalReportReady(all_segments=final_content)
        
        # If not all segments are ready, return None to wait for more SegmentApproved events.
        return None

    @step
    async def create_and_distribute_report(self, ev: FinalReportReady) -> StopEvent:
        # This is the final step in the high-score path.
        print(f"Step 5: Creating and distributing final report...")
        #... logic to call a ReportFormattingTool and EmailDistributionTool...
        full_memo_content = "\n\n".join(ev.all_segments)
        print("--- FINAL MEMO ---")
        print(full_memo_content)
        print("--------------------")
        print("Report generated in DOCX, PDF, and sent via email.")
        
        # Return a StopEvent to end the workflow, passing the final result.
        return StopEvent(result={"status": "complete", "content": full_memo_content})

# To run the workflow:
# async def main():
#     workflow = MemoProcessingWorkflow()
#     result = await workflow.run(start_event=StartEvent(some_initial_data="..."))
#     print(f"Workflow finished with result: {result}")
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
```

-----

## Section 5: Implementing the Conditional Paths

To keep the workflow logic clean and modular, the specific actions taken in each conditional path (regeneration, notification, file conversion) are abstracted into **Tools**. The workflow steps then act as orchestrators, deciding which tool to call with what parameters based on the incoming event. This is a standard and robust design pattern for agentic systems.³⁴

### 5.1 The Low-Score Path: Automated Regeneration

  * **Trigger:** An `EvaluationComplete` event where the composite score is below a predefined threshold (e.g., `< 2.5`).
  * **Action:** The `route_based_on_score` step emits a `RegenerationRequired` event. The `handle_regeneration` step listens for this event and invokes a `RegenerationTool`.
  * **Tool Implementation:** The `RegenerationTool` is a powerful function that can employ a cascade of strategies to improve the output. It might start with the cheapest option and escalate if quality does not improve.
      * **Prompt Refinement:** The tool can analyze the "reasoning" fields in the evaluation scores to identify the failure mode (e.g., hallucination, missed facts) and dynamically rewrite the original generation prompt to be more specific or to include negative constraints.
      * **Context Expansion:** If the issue was "Missing Out," the tool can re-run the re-ranking stage to retrieve a larger context (e.g., top 16 chunks instead of top 8) and attempt generation again with more information.
      * **Model Escalation:** If simpler methods fail, the tool can escalate the task to a more powerful and expensive model, such as OpenAI's `o1-mini` or Google's `Gemini 2.5 Pro`, which are specialized for complex reasoning and can often overcome the failures of smaller models.

The output of this tool is a new, regenerated text, which is then wrapped in a new `SegmentGenerated` or `EvaluationComplete` event to re-enter the evaluation loop. This creates a closed-loop, self-correcting system.

### 5.2 The Medium-Score Path: Human-in-the-Loop (HITL)

  * **Trigger:** An `EvaluationComplete` event with a score in the medium range (e.g., `2.5 <= score < 4.5`).
  * **Action:** The workflow emits a `HumanReviewRequired` event. The `handle_human_review` step listens for this event and calls a `SlackNotificationTool`.
  * **Tool Implementation:** This tool integrates with communication platforms like Slack or Discord to bring a human expert into the decision-making process. Using a library like `slack_sdk`, the tool can be implemented as follows.³⁵

<!-- end list -->

````python
import os
import json
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from llama_index.core.tools import FunctionTool
from typing import List, Dict, Any

def send_for_human_review_slack(
    segment_id: str,
    segment_text: str,
    evaluation_scores: Dict[str, Any],
    context_chunks: List[str]
) -> str:
    """
    Sends a message to a Slack channel for human review and approval.
    In a real system, this would be asynchronous and wait for a callback.
    """
    try:
        client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
        channel_id = os.environ.get("SLACK_CHANNEL_ID")
        
        # Format the context and scores for readability in Slack
        context_preview = "\n> ".join(context_chunks[:2]) + "\n>..."
        scores_preview = json.dumps(evaluation_scores, indent=2)
        
        message_blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Human Review Required: {segment_id}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Generated Text:*\n" + segment_text
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Evaluation Scores:*\n```{scores_preview}```"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Source Context Preview:*\n{context_preview}"
                    }
                ]
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "value": f"approve_{segment_id}"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Reject"},
                        "style": "danger",
                        "value": f"reject_{segment_id}"
                    }
                ]
            }
        ]
        
        response = client.chat_postMessage(
            channel=channel_id,
            blocks=message_blocks,
            text=f"Review required for {segment_id}" # Fallback text for notifications
        )
        
        # In a real application, the workflow would pause here and wait for an
        # incoming webhook from Slack triggered by the button clicks.
        # For this example, we simulate an immediate approval.
        return "human_approval_initiated"
        
    except (SlackApiError, KeyError) as e:
        error_message = e.response['error'] if isinstance(e, SlackApiError) else "Missing SLACK_BOT_TOKEN or SLACK_CHANNEL_ID env variables."
        print(f"Error sending Slack message: {error_message}")
        return "slack_notification_failed"

# Wrap the function as a LlamaIndex tool
human_review_tool = FunctionTool.from_defaults(fn=send_for_human_review_slack)
````

### 5.3 The High-Score Path: Finalization and Distribution

  * **Trigger:** An `EvaluationComplete` event with a high score (e.g., `>= 4.5`) or a manual approval from the HITL workflow.
  * **Action:** The workflow emits a `SegmentApproved` event. A final aggregation step within the workflow collects all approved segments. Once all segments for a memo are collected, it emits a `FinalReportReady` event, which triggers a `FinalizeAndDistributeTool`.
  * **Tool Implementation:** This tool handles the final assembly and output. A robust and flexible approach is to first generate a single, well-structured HTML document as a master format. This master HTML can then be reliably converted into various other formats, simplifying the pipeline.

#### Python Code for File Conversion

The following snippets demonstrate how to convert the final aggregated text into different document formats.

##### 1\. Aggregation to HTML (Master Format)

```python
def aggregate_to_html(segments: List[str]) -> str:
    """Aggregates text segments into a simple HTML document."""
    body_content = "".join([f"<p>{segment}</p>" for segment in segments])
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Generated Financial Memo</title>
        <style>
            body {{ font-family: sans-serif; line-height: 1.6; }}
            h1 {{ color: #333; }}
        </style>
    </head>
    <body>
        <h1>Financial Memo</h1>
        {body_content}
    </body>
    </html>
    """
    return html_template
```

##### 2\. Conversion to DOCX (using `python-docx`)

```python
# Requires: pip install python-docx
from docx import Document

def convert_text_to_docx(text_content: str, output_filename: str):
    """Converts a string of text into a .docx file."""
    document = Document()
    document.add_heading('Financial Memo', 0)
    
    for paragraph in text_content.split('\n\n'):
        document.add_paragraph(paragraph)
        
    document.save(output_filename)
    print(f"Successfully saved DOCX to {output_filename}")

# Usage:
# final_text = "\n\n".join(approved_segments_list)
# convert_text_to_docx(final_text, "financial_memo.docx")
```

³⁶

##### 3\. Conversion to PDF (using `fpdf2`)

```python
# Requires: pip install fpdf2
from fpdf import FPDF

def convert_text_to_pdf(text_content: str, output_filename: str):
    """Converts a string of text into a .pdf file."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Use multi_cell for automatic line breaking
    pdf.multi_cell(0, 10, txt=text_content)
    
    pdf.output(output_filename)
    print(f"Successfully saved PDF to {output_filename}")

# Usage:
# final_text = "\n\n".join(approved_segments_list)
# convert_text_to_pdf(final_text, "financial_memo.pdf")
```

³⁸

##### 4\. Conversion to Slides/PPTX (using `aspose.slides`)

This library is particularly useful as it can directly ingest HTML, simplifying the process.

```python
# Requires: pip install aspose.slides
import aspose.slides as slides

def convert_html_to_pptx(html_content: str, output_filename: str):
    """Converts an HTML string into a .pptx presentation."""
    with slides.Presentation() as pres:
        # Aspose.Slides can add content directly from an HTML string
        pres.slides.add_from_html(html_content)
        pres.save(output_filename, slides.export.SaveFormat.PPTX)
    print(f"Successfully saved PPTX to {output_filename}")

# Usage:
# html_string = aggregate_to_html(approved_segments_list)
# convert_html_to_pptx(html_string, "financial_memo.pptx")
```

³⁹

After generation, standard Python libraries like `smtplib` and `email` can be used to attach these files and distribute the final report.

### 5.4 Agentic Workflow Decision Matrix

The logic of the entire agentic system can be summarized in a clear decision matrix. This serves as a crucial piece of documentation for understanding, debugging, and extending the system's behavior by making the complex conditional logic explicit.

#### Table 4: Agentic Workflow Decision Matrix

| Composite Score Range | Triggered Event | Workflow Step | Tool(s) Called | Downstream Action |
| :--- | :--- | :--- | :--- | :--- |
| **\< 2.5 (Low)** | `RegenerationRequired` | `handle_regeneration` | `RegenerationTool` | Re-run generation with enhanced parameters/model and re-evaluate. |
| **2.5 - 4.4 (Medium)** | `HumanReviewRequired` | `handle_human_review` | `SlackNotificationTool` | Pause workflow and send an alert to a human expert for approval/rejection. |
| **\>= 4.5 (High)** | `SegmentApproved` | `aggregate_and_finalize` | `ReportFormattingTool`, `EmailDistributionTool` | Aggregate segments, convert to DOCX/PDF/PPTX/HTML, and distribute. |

-----

## Conclusion

This report has detailed a comprehensive, end-to-end framework for building a production-grade RAG system for financial memo generation. The architecture is founded on the principle that true reliability in AI systems stems from a deeply integrated cycle of rigorous evaluation and intelligent, automated action. By moving beyond simplistic metrics and static pipelines, this framework establishes a new standard for developing trustworthy AI in domains where the cost of error is high.

The multi-faceted evaluation strategy provides a complete picture of system performance. At the retrieval stage, it combines set-based and rank-aware metrics to ensure that the context provided to the LLM is both complete (high recall) and precise (high MAP/NDCG). For generation, the LLM-as-a-Judge, trained on distilled human expertise, offers a scalable and nuanced assessment across critical axes like groundedness, content coverage, and confidence calibration. This quantitative foundation, combined with a protocol for ensuring cross-model consistency, enables a data-driven approach to optimizing every component of the system.

The agentic post-processing workflow, built on LlamaIndex, translates these evaluations into autonomous action. It creates a self-correcting system that can identify and fix its own errors through automated regeneration, intelligently escalate ambiguous cases for human review, and confidently finalize and distribute high-quality outputs. This conditional, state-driven approach represents the future of reliable AI, where systems are not just passive generators but active participants in ensuring the quality and safety of their own work.

Ultimately, this framework is more than a solution for the TinyRAG project; it is a blueprint for the next generation of AI systems. As organizations increasingly deploy LLMs for mission-critical tasks, the methodologies for ensuring accuracy, verifiability, and reliability will become the primary determinants of success. Engineers and data scientists are encouraged to adapt, extend, and build upon this framework, contributing to an ecosystem of tools and practices that will make powerful AI systems not just capable, but also accountable.

---

**🎉 Congratulations! You've Completed the RAG Implementation Series!**  
← **Previous:** [A Framework for Rigorous Evaluation and Agentic Post-Processing](/post/framework-rigorous-evaluation-agentic-postprocessing-financial-rag)  
**Series Complete!** You've mastered production-ready RAG systems from business case to autonomous workflows.  
📋 **[Return to Learning Path](/knowledge/rag)** | **Progress: 9/9 Complete** 🏆✅✅✅✅✅✅✅✅✅  

**What's Next?** Explore our [Data Science Foundations](/knowledge/foundations) for mathematical and statistical deep dives, or start building your own production RAG system!