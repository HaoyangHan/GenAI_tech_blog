---
title: "A Framework for Rigorous Evaluation and Agentic Post-Processing in Financial RAG Systems"
category: "Evaluation"
date: "July 27, 2025"
summary: "This post presents a production-grade evaluation framework for financial RAG systems, detailing a multi-layered, quantitative approach to retrieval and generation assessment, and introducing agentic post-processing for robust, auditable outputs."
slug: "framework-rigorous-evaluation-agentic-postprocessing-financial-rag"
tags: ["RAG", "Evaluation", "LLM", "Ragas", "Financial AI", "Metrics", "Agentic Workflow", "LlamaIndex"]
author: "Haoyang Han"
---

**📚 RAG Implementation Series - Article 8 of 9:**  
[Complete Learning Path](/knowledge/rag) | ← [Previous: Retrieval Strategies](/post/retrieval-strategies-in-financial-rag) | **Current: Evaluation Framework** → [Next: Agentic Workflows](/post/agentic-post-processing-tinyrag)

# A Framework for Rigorous Evaluation and Agentic Post-Processing in Financial RAG Systems


# A Framework for Rigorous Evaluation and Agentic Post-Processing in Financial RAG Systems

## Introduction

The proliferation of Retrieval-Augmented Generation (RAG) systems has marked a significant milestone in applied artificial intelligence. While early proof-of-concept pipelines demonstrated the potential of grounding large language model (LLM) outputs in external knowledge, the transition to production-grade systems, particularly in high-stakes domains like finance, presents a far greater challenge. In the context of generating financial memoranda, where precision, factual accuracy, and auditable groundedness are not merely desirable features but fundamental requirements, the standard RAG architecture is insufficient. A system operating in this environment must be robust, reliable, and capable of self-correction.

This report introduces a comprehensive framework for the "TinyRAG" project, a system designed to automate the generation of financial memos. It moves beyond simplistic evaluations to propose a multi-faceted, rigorous methodology for measuring performance at every stage of the pipeline—from initial document retrieval to final text generation. The framework is built upon a **"measure, verify, and act"** loop, ensuring that every component's quality is quantified before its output proceeds. This approach is critical for building a system that is not only powerful but also trustworthy.

The analysis is presented in two main parts. **Part I** details a multi-layered evaluation strategy. It begins with a deep dive into quantifying the performance of the retrieval funnel, using both classic and rank-aware information retrieval metrics to assess the initial candidate selection and the subsequent re-ranking stage. It then architects a sophisticated, scalable **LLM-as-a-Judge** system, which learns from human expert patterns to evaluate generated content against a granular set of criteria including hallucination, content coverage, and confidence calibration. This part concludes by establishing a protocol for benchmarking a diverse portfolio of commercial and open-source LLMs to ensure consistent quality and strategic flexibility.

**Part II** transitions from evaluation to action, outlining a blueprint for an **agentic post-processing workflow**. This intelligent system, built using the LlamaIndex framework, is designed to autonomously interpret the quantitative evaluation scores from Part I. Based on these scores, the agent makes decisions, triggering conditional pathways for automated content regeneration, escalation for human-in-the-loop review, or finalization and distribution of the report in various formats. This framework represents a significant step toward creating AI systems that are not just generative, but also accountable and self-aware.

-----

# Part I: A Multi-Faceted Evaluation Strategy for Production RAG

## Section 1: Evaluating the Retrieval Funnel: From Candidates to Context

The retrieval component is the foundation of any RAG system. Its ability to surface the correct and most relevant information from a vast corpus directly determines the quality ceiling of the final generated output. In the TinyRAG system, retrieval is not a monolithic step but a two-stage funnel designed to balance the competing needs of comprehensiveness and precision.

### 1.1 The Two-Stage Retrieval Problem in TinyRAG

The architecture employs a two-stage retrieval process, a common and highly effective pattern in modern information retrieval systems.

  * **Stage 1: Candidate Generation (Broad Recall):** The first stage utilizes a fast and computationally inexpensive retrieval method (e.g., dense vector search with a standard embedding model) to cast a wide net. From the entire document corpus, it retrieves a large set of candidate chunks—for instance, the top 100 most likely candidates. The primary objective of this stage is to maximize **recall**, ensuring that all potentially relevant pieces of information are captured and passed to the next stage.
  * **Stage 2: Re-ranking (Focused Precision):** The second stage takes the 100 candidates from Stage 1 and applies a more sophisticated, and often more computationally expensive, re-ranking model. This model's purpose is to meticulously analyze the relationship between the query and each candidate chunk, re-ordering the list to place the most relevant items at the very top. From this re-ranked list, a much smaller set—typically the top 8 or 16 chunks—is selected to form the final context that will be fed to the LLM. The objective here is to maximize **precision** and **ranking quality** in this final context set.

Evaluating these two stages requires distinct methodologies, as they are optimized for different outcomes.

### 1.2 Stage 1 Evaluation: Assessing the Candidate Pool (k=100)

The evaluation of the first stage is predicated on a simple question: given a set of 8 "golden" chunks hand-labeled by a human financial expert as essential for a given memo, how well did our initial retrieval of 100 candidates capture them? This can be framed as a binary classification problem for each chunk in the corpus.

#### Metrics and Interpretation

For this stage, we employ a set of standard, set-based metrics to measure performance.¹

  * **Precision@100:** This metric answers the question, "Of the 100 chunks our first-stage retriever found, what fraction were actually relevant?" It is a measure of the signal-to-noise ratio in the candidate pool. While a high score is good, a low score is not necessarily catastrophic, as the re-ranker is designed to filter out noise. The formula is $Precision@K = \\frac{\\text{Number of relevant items in K}}{K}$.
  * **Recall@100:** This is the most critical metric for Stage 1. It answers, "Of the 8 total relevant chunks that exist, what fraction did we find in our initial pool of 100?" The formula is $Recall@K = \\frac{\\text{Number of relevant items in K}}{\\text{Total number of relevant items}}$.² The primary goal of this first retrieval stage is not perfection but completeness. Low precision can be tolerated and subsequently corrected by the re-ranker, but low recall represents an unrecoverable error. If a crucial piece of information—one of the 8 expert-identified chunks—is not present in the initial 100 candidates, it is lost to the rest of the pipeline. The re-ranker and the LLM will never have the opportunity to see it. Therefore, the **Recall@100** score establishes the absolute maximum quality ceiling for the entire RAG system for that specific query. All engineering efforts for this stage, such as the choice of embedding model, chunking strategy, or query expansion techniques, should be obsessively optimized to maximize this metric.
  * **F1-Score@100:** As the harmonic mean of precision and recall, the F1-score provides a single, balanced measure of the retriever's overall performance. It is particularly useful when comparing different retriever configurations where one might improve precision at the cost of recall, or vice-versa.⁴ The formula is $F1 = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$.⁵

#### Python Implementation

The following Python code provides a straightforward implementation for calculating these metrics for a single query. It uses basic set operations for efficiency and clarity.

```python
from typing import List, Set, Dict, Any

def calculate_retrieval_metrics_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> Dict[str, float]:
    """
    Calculates Precision@k, Recall@k, and F1-Score@k for a single query.

    Args:
        retrieved_ids: An ordered list of chunk IDs retrieved by the system.
        relevant_ids: A set of chunk IDs labeled as relevant by human experts.
        k: The cutoff for the number of retrieved items to consider.

    Returns:
        A dictionary containing precision, recall, and f1-score at k.
    """
    # Ensure we only consider the top k retrieved items
    top_k_retrieved = set(retrieved_ids[:k])

    # Calculate the number of relevant items among the top k
    true_positives = len(top_k_retrieved.intersection(relevant_ids))

    # Precision@k: TP / (TP + FP) = TP / k
    precision_at_k = true_positives / k if k > 0 else 0.0

    # Recall@k: TP / (TP + FN) = TP / total_relevant_items
    total_relevant = len(relevant_ids)
    recall_at_k = true_positives / total_relevant if total_relevant > 0 else 0.0

    # F1-Score@k: 2 * (Precision * Recall) / (Precision + Recall)
    if precision_at_k + recall_at_k == 0:
        f1_at_k = 0.0
    else:
        f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

    return {
        f"precision@{k}": precision_at_k,
        f"recall@{k}": recall_at_k,
        f"f1_score@{k}": f1_at_k,
    }

# --- Example Usage for TinyRAG Stage 1 ---
# A human expert has identified 8 chunks as being the most important.
expert_labeled_relevant_chunks = {"chunk_5", "chunk_12", "chunk_23", "chunk_42", "chunk_55", "chunk_67", "chunk_89", "chunk_99"}

# Our Stage 1 retriever returns a list of 100 candidate chunk IDs.
# Let's assume it found 7 of the 8 relevant chunks (it missed chunk_99).
# We'll create a list of 100 chunks, mixing irrelevant and relevant ones.
retrieved_found = ["chunk_5", "chunk_12", "chunk_23", "chunk_42", "chunk_55", "chunk_67", "chunk_89"]
retrieved_irrelevant = [f"irrelevant_chunk_{i}" for i in range(93)]
stage1_retrieved_candidates = retrieved_found + retrieved_irrelevant # Total 100 chunks.

# Evaluate the performance of the Stage 1 retriever
stage1_metrics = calculate_retrieval_metrics_at_k(
    retrieved_ids=stage1_retrieved_candidates,
    relevant_ids=expert_labeled_relevant_chunks,
    k=100
)

print("--- Stage 1 Retrieval Evaluation (k=100) ---")
print(f"Precision@100: {stage1_metrics['precision@100']:.4f}")
print(f"Recall@100: {stage1_metrics['recall@100']:.4f}")
print(f"F1-Score@100: {stage1_metrics['f1_score@100']:.4f}")
```

### 1.3 Stage 2 Evaluation: Quantifying Re-Ranker Performance (k=8 or 16)

The re-ranker's task is fundamentally different from the initial retriever's. It does not discover new information; its universe is confined to the 100 candidates provided by Stage 1. Its sole purpose is to improve the quality of the top-K results that will form the LLM's context by pushing the most relevant chunks to the highest ranks.

This distinction renders simple, set-based metrics like Precision@K and Recall@K insufficient for a complete evaluation. While we can and should calculate Precision@8 and Recall@8, these metrics are rank-agnostic. They would produce the same score whether a critical chunk appeared at rank 1 or rank 8.³ This failure to account for ordering misses the entire point of re-ranking. A system that places the most vital information at the top of the context is demonstrably superior to one that buries it. To capture this crucial aspect of performance, we must employ rank-aware metrics from the field of Information Retrieval.

#### Rank-Aware Metrics and Interpretation

  * **Mean Average Precision (MAP):** MAP provides a single-figure score that heavily rewards systems for placing relevant documents at the top of the ranked list.⁶ It is calculated by averaging the precision scores obtained at each point a relevant document is retrieved. A higher MAP score indicates that, on average, relevant documents are found earlier in the search results, which is precisely the behavior we want from our re-ranker.²
  * **Normalized Discounted Cumulative Gain (NDCG):** NDCG is an even more powerful and flexible metric. It operates on two principles: 1) highly relevant documents are more valuable than marginally relevant ones, and 2) the value of a relevant document diminishes the further down the list it appears (a "discount" factor).⁶ NDCG is particularly valuable because it supports graded relevance. For TinyRAG, this means we could have experts label chunks not just as "relevant" (1) or "not relevant" (0), but on a scale, e.g., "critically important" (3), "highly relevant" (2), "somewhat relevant" (1). NDCG would then measure how well the re-ranker prioritizes the most critical information.

The re-ranker's performance is also constrained by the output of Stage 1. The total number of relevant documents available to the re-ranker is only the subset of the 8 "golden" chunks that were successfully found in the initial 100 candidates. The re-ranker cannot discover new relevant documents; it can only work with what it's given. Therefore, its primary value lies in dramatically boosting the precision and ranking quality (as measured by MAP and NDCG) of the final context fed to the LLM, thereby improving the signal-to-noise ratio and enabling more accurate generation.

#### Python Implementation

For rank-aware metrics, using a specialized library like `pytrec_eval` is the standard and most reliable approach. It is a Python interface to the official TREC evaluation tool, ensuring correctness and consistency.

```python
import pytrec_eval
from typing import Dict, List, Set

def evaluate_reranker_performance(
    reranked_ids_with_scores: Dict[str, float],
    relevant_ids_with_grades: Dict[str, int],
    k: int
) -> Dict[str, float]:
    """
    Evaluates reranker performance using rank-aware metrics (MAP and NDCG).

    Args:
        reranked_ids_with_scores: A dictionary of {chunk_id: score} from the reranker.
        relevant_ids_with_grades: A dictionary of {chunk_id: relevance_grade} from experts.
                                   Grades are integers (e.g., 0=not relevant, 1=relevant, 2=highly relevant).
        k: The cutoff for evaluation (e.g., 8 or 16).

    Returns:
        A dictionary containing MAP@k and NDCG@k scores.
    """
    # pytrec_eval expects a specific format:
    # qrels: ground truth {query_id: {doc_id: relevance_grade}}
    # run: system output {query_id: {doc_id: score}}
    
    query_id = 'financial_memo_q1'
    
    qrels = {query_id: relevant_ids_with_grades}
    run = {query_id: reranked_ids_with_scores}

    # Define the metrics to calculate. `map_cut` and `ndcg_cut` are the @K versions.
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, 
        {f'map_cut_{k}', f'ndcg_cut_{k}'}
    )

    results = evaluator.evaluate(run)
    
    # Extract the scores for our single query
    query_results = results.get(query_id, {})
    
    return {
        f"map@{k}": query_results.get(f'map_cut_{k}', 0.0),
        f"ndcg@{k}": query_results.get(f'ndcg_cut_{k}', 0.0),
    }

# --- Example Usage for TinyRAG Stage 2 ---
# The same 8 relevant chunks, but now with relevance grades.
# Let's say chunks 42 and 99 are critically important (grade 2).
expert_labeled_graded_chunks = {
    "chunk_5": 1, "chunk_12": 1, "chunk_23": 1, "chunk_42": 2,
    "chunk_55": 1, "chunk_67": 1, "chunk_89": 1, "chunk_99": 2
}

# The reranker takes the 100 candidates and outputs a new ranked list with scores.
# A good reranker places the relevant items (especially high-grade ones) at the top.
# Note: The reranker can only rank items that were in the initial 100 candidates.
# In our example, chunk_99 was missed, so it cannot appear here.
reranker_output = {
    "chunk_42": 0.98,  # Correctly ranked the most important item first
    "chunk_5": 0.95,
    "chunk_89": 0.91,
    "chunk_12": 0.85,
    "chunk_irrelevant_1": 0.82,
    "chunk_55": 0.76,
    "chunk_23": 0.71,
    "chunk_irrelevant_2": 0.65,
    "chunk_67": 0.60,  # Found all 7 available relevant chunks
    #... other chunks up to 100
}

# Also calculate traditional metrics for the top 8 reranked items
reranked_ids_list = list(reranker_output.keys())
stage2_set_metrics = calculate_retrieval_metrics_at_k(
    retrieved_ids=reranked_ids_list,
    relevant_ids=set(expert_labeled_graded_chunks.keys()),
    k=8
)

# And evaluate with rank-aware metrics
stage2_rank_metrics = evaluate_reranker_performance(
    reranked_ids_with_scores=reranker_output,
    relevant_ids_with_grades=expert_labeled_graded_chunks,
    k=8
)

print("\n--- Stage 2 Reranker Evaluation (k=8) ---")
print(f"Precision@8: {stage2_set_metrics['precision@8']:.4f}")
print(f"Recall@8: {stage2_set_metrics['recall@8']:.4f}")
print(f"F1-Score@8: {stage2_set_metrics['f1_score@8']:.4f}")
print(f"MAP@8: {stage2_rank_metrics['map@8']:.4f}")
print(f"NDCG@8: {stage2_rank_metrics['ndcg@8']:.4f}")
```

#### Table 1: Retrieval Evaluation Metrics Summary

To ensure clarity and consistency across the data science team, the following table summarizes the key retrieval metrics, their definitions, and their specific role within the TinyRAG evaluation framework.

| Metric | Formula/Definition | Stage Applied | Interpretation for TinyRAG |
| :--- | :--- | :--- | :--- |
| **Precision@K** | `#(Relevant Retrieved in K) / K` | Stage 1 & 2 | Measures the signal-to-noise ratio in the retrieved set. High precision in Stage 2 is crucial for clean context. |
| **Recall@K** | `#(Relevant Retrieved in K) / #(Total Relevant)` | Stage 1 & 2 | Measures how many of the essential facts were found. This is the single most important metric for Stage 1. |
| **F1-Score@K** | `2 * (P@K * R@K) / (P@K + R@K)` | Stage 1 & 2 | A balanced measure of precision and recall, useful for overall system comparison. |
| **MAP@K** | `Mean of Average Precision scores across queries` | Stage 2 | Rewards ranking relevant chunks higher. The primary metric for assessing re-ranker quality. |
| **NDCG@K** | `DCG / IDCG` | Stage 2 | Rewards ranking *more important* chunks higher. The gold standard, especially if we introduce graded relevance. |

-----

## Section 2: Deconstructing Generation Quality with an LLM-as-a-Judge

Once the retrieval funnel has produced a high-quality context, the focus shifts to the LLM's generation quality. Evaluating generated text, especially for nuanced financial memos, is notoriously difficult. Simple lexical metrics like ROUGE or BLEU are inadequate as they fail to capture factual correctness, semantic meaning, or logical coherence. The most reliable method is human evaluation, but it is slow, expensive, and unscalable.

The solution is a hybrid approach that leverages human expertise to create a scalable, automated evaluation system: the **LLM-as-a-Judge**.

### 2.1 From Human Expertise to Scalable Evaluation

The process of building a trustworthy LLM judge is methodical and occurs in two distinct phases:

1.  **Phase 1: Ground Truth Creation and Rubric Distillation:** The foundation of the entire system is built on genuine domain expertise. In this phase, a cohort of human financial experts evaluates an initial set of 50+ generated memos. Crucially, they do not just provide a single score; they provide detailed critiques and justifications for their ratings against a predefined set of criteria.⁸ This rich, qualitative feedback is then analyzed to distill a highly specific, unambiguous scoring rubric. This step forces clarity and ensures the evaluation criteria are well-defined before any automation begins.
2.  **Phase 2: Few-Shot Judge Training:** With the expert-validated rubric and a set of high-quality, annotated examples, we can now "train" our LLM-as-a-Judge. This is not fine-tuning in the traditional sense, but rather in-context learning via few-shot prompting.¹¹ The detailed rubric and the expert-annotated examples are embedded directly into the judge's prompt, showing it exactly how to perform the evaluation task. This method allows the judge to learn the complex patterns of expert assessment from a small number of examples.¹³

### 2.2 Architecting the LLM-as-a-Judge Prompt

A simple numeric score from a judge is uninformative and unactionable. To create a powerful diagnostic tool, the judge must provide a structured, multi-faceted evaluation. Therefore, the prompt is designed to elicit a detailed JSON object as output. This object contains a separate score and a textual justification for each of the five core evaluation criteria. This structured output is not only more informative for human review but is also machine-readable, enabling its direct use by the agentic post-processing workflow described in Part II.

The master prompt for the LLM-as-a-Judge is meticulously crafted to include several key components:

  * **Role-Playing:** The prompt begins by assigning a specific, expert persona to the LLM, such as, "You are a meticulous financial analyst and senior editor at a top-tier investment bank. Your task is to critically evaluate the quality of a generated memo segment".¹⁵
  * **Task Definition:** It clearly outlines the inputs (source context chunks, a human-written reference memo, and the AI-generated memo segment) and the goal of the evaluation.
  * **Detailed Scoring Rubric:** The prompt explicitly defines the 1-to-5 scoring scale for each of the five evaluation criteria. This removes ambiguity and ensures consistent scoring.¹⁵
  * **Few-Shot Examples:** The prompt includes 3-5 high-quality examples derived from the human evaluation phase. Each example shows the inputs, the desired JSON output, and the expert-level reasoning for each score, demonstrating how to handle nuance and edge cases.¹¹
  * **Output Format Enforcement:** The prompt concludes with a strict instruction to produce a single, valid JSON object and nothing else, often specifying the JSON schema to ensure reliable parsing.¹⁷

### 2.3 Implementing the Five Evaluation Criteria

The following sections detail the implementation and scoring logic for each of the five criteria identified for TinyRAG.

#### A. Groundedness & Hallucination

  * **Definition:** This criterion measures the factual consistency of the generated text against the provided source context chunks. A hallucination is defined as any verifiable fact, number, or claim in the generated output that does not exist in or cannot be directly inferred from the source context.¹⁸
  * **Implementation:** A prompt-based hallucination detector is the most effective approach. The LLM-as-a-Judge is instructed to perform a sentence-by-sentence cross-reference, comparing each claim in the generated text with the information present in the source chunks. The prompt used for this sub-task is heavily inspired by proven techniques for this problem.¹⁸
  * **Python Code:** The following function encapsulates the call to the LLM judge for a hallucination score. It constructs a specific part of the master prompt focused on this criterion.

<!-- end list -->

```python
import os
import json
from openai import OpenAI
from typing import Dict, Any

# It is assumed that the OpenAI API key is set as an environment variable
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_llm_judge_evaluation(
    context: str, 
    generated_text: str, 
    human_reference: str, 
    judge_model: str = "gpt-4.1-mini"
) -> Dict[str, Any]:
    """
    Calls an LLM-as-a-Judge to evaluate a generated text against multiple criteria.

    Args:
        context: The source context chunks provided to the generator LLM.
        generated_text: The output from the generator LLM.
        human_reference: The human-written "golden" memo for comparison.
        judge_model: The model to use as the judge.

    Returns:
        A dictionary containing the structured evaluation results.
    """
    
    # The full prompt is extensive and includes the detailed rubric and few-shot examples.
    # This is a condensed representation of the core instruction.
    system_prompt = """
    You are a meticulous financial analyst and senior editor. Your task is to evaluate an AI-generated memo segment.
    You will be provided with the source 'Context', a 'Human-Written Reference', and the 'AI-Generated Text'.
    
    Evaluate the 'AI-Generated Text' based on the following five criteria, using the provided rubric.
    For each criterion, provide a score from 1 to 5 and a brief, precise justification for your score.
    Your final output MUST be a single, valid JSON object with no other text.
    
    The JSON structure must be:
    {
      "hallucination": {"score": <int>, "reasoning": "<text>"},
      "captured_rate": {"score": <int>, "reasoning": "<text>"},
      "value_add": {"score": <int>, "reasoning": "<text>"},
      "missing_out": {"score": <int>, "reasoning": "<text>"},
      "confidence_calibration": {"score": <int>, "reasoning": "<text>"}
    }
    
    --- SCORING RUBRIC ---
    1. Hallucination:
       - 5: Fully grounded. All facts/numbers are directly verifiable in the source context.
       - 3: Mostly grounded, but may contain minor, non-critical inferences. No factual errors.
       - 1: Contains significant factual errors or fabricated information not in the context.
    
    2. Captured Rate (vs. Human Reference):
       - 5: Captures all critical and most secondary points from the human reference.
       - 3: Captures all critical points but may miss some secondary details.
       - 1: Fails to capture one or more critical points from the human reference.
       
    3. Value Add (Novel Insights vs. Human Reference):
        - 5: Surfaces a non-obvious, highly relevant insight by synthesizing information from multiple chunks that provides a new perspective.
        - 3: Surfaces a minor, useful fact or connection that was not in the human reference.
        - 1: Provides no new information beyond what the human expert captured.
        
    4. Missing Out (vs. Human Reference):
        - 5: No critical or secondary information from the human reference is missing.
        - 3: A few minor, non-essential details from the human reference are omitted.
        - 1: Omits critical information that was present in the human reference.
        
    5. Confidence Calibration (Assessing self-reported confidence tags like <CONFIDENCE: HIGH>):
        - 5: Confidence tags are perfectly calibrated. Reports HIGH on fully grounded facts and LOW/MEDIUM on inferred points.
        - 3: Confidence tags are mostly reasonable but may be slightly over/under-confident.
        - 1: Confidence is miscalibrated. Reports HIGH confidence on hallucinated facts.
        
    --- FEW-SHOT EXAMPLES ---
    [... Here, 3-5 detailed, expert-annotated examples would be inserted...]
    """
    
    user_prompt = f"""
    --- START OF DATA ---
    <CONTEXT>:
    {context}
    
    <HUMAN_REFERENCE>:
    {human_reference}
    
    <AI_GENERATED_TEXT>:
    {generated_text}
    --- END OF DATA ---
    
    Now, provide your evaluation in the specified JSON format.
    """
    
    try:
        # In a real implementation, use a robust LLM client library.
        # This is a simplified representation.
        # response = client.chat.completions.create(
        #     model=judge_model,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt}
        #     ],
        #     response_format={"type": "json_object"},
        #     temperature=0.0
        # )
        # parsed_response = json.loads(response.choices[0].message.content)
        
        # For demonstration, we return a mock response.
        mock_response_str = """
        {
          "hallucination": {"score": 4, "reasoning": "The text is well-grounded, but it combines two figures from the context into a total that isn't explicitly stated, which is a minor inference."},
          "captured_rate": {"score": 5, "reasoning": "All key financial metrics mentioned in the human reference were correctly identified and included."},
          "value_add": {"score": 2, "reasoning": "It noted a small percentage change year-over-year that the human expert omitted, which is a minor but useful addition."},
          "missing_out": {"score": 5, "reasoning": "No significant information from the human reference was left out."},
          "confidence_calibration": {"score": 3, "reasoning": "The model correctly reported HIGH confidence on direct figures, but also reported HIGH on the inferred total, where MEDIUM would have been more appropriate."}
        }
        """
        parsed_response = json.loads(mock_response_str)
        return parsed_response
    except Exception as e:
        print(f"Error during LLM-as-a-Judge evaluation: {e}")
        return None
```

#### B. Content Coverage (Captured Rate vs. Missing Out)

  * **Definition:** These two criteria are complementary and assess the semantic overlap between the AI-generated memo and the human-written reference. **Captured Rate** measures how much of the essential human-written content the AI successfully included, while **Missing Out** focuses on the significance of what was omitted.
  * **Implementation:** The LLM judge is given both texts and instructed to first deconstruct the human reference into a list of key semantic points or claims. It then verifies the presence of each point in the AI-generated text.
  * **Scoring:** The Captured Rate can be quasi-quantitative ((Key points present in both) / (Total key points in human reference)), but the final score is a qualitative assessment on the 1-5 scale based on the importance of the captured points. The Missing Out score is inherently qualitative, focusing on the impact of the omitted information.

#### C. Novelty & Value-Add

  * **Definition:** This is a sophisticated metric designed to reward the RAG system for one of its key potential advantages: synthesis. It measures whether the AI system identified and surfaced new, valuable, and grounded insights from the source context that the human expert did not include in their reference memo.
  * **Implementation:** This involves a reverse-check. The judge identifies key claims in the AI output that are grounded in the source context but are absent from the human reference. It then assesses the importance of this novel information.
  * **Scoring:** A purely qualitative score based on the significance of the new insight. A score of 5 would be reserved for a non-obvious connection synthesized from multiple disparate source chunks that provides a genuinely new perspective.

#### D. Confidence Estimation & Calibration

  * **Definition:** This criterion evaluates the reliability of the generator LLM's own self-awareness. It is not enough for the model to be correct; a trustworthy system should also know when it might be incorrect.
  * **Implementation:** This is implemented as an integrated, two-step process.
    1.  **Generation with Confidence Tags:** The prompt for the generator LLM is modified to instruct it to interleave confidence tags (e.g., `<CONFIDENCE: HIGH>`, `<CONFIDENCE: MEDIUM>`, `<CONFIDENCE: LOW>`) within its output for key statements.
    2.  **Judge Calibration Check:** The LLM-as-a-Judge receives the text with these tags. A specific instruction in its prompt directs it to assess the appropriateness of these tags. It compares its own groundedness finding for a statement with the generator's self-reported confidence. For example, if the generator reports `<CONFIDENCE: HIGH>` for a statement the judge identifies as a hallucination, it will receive a very low calibration score. This creates a powerful meta-evaluation metric for tracking the model's reliability over time.²⁰

#### Table 2: LLM-as-a-Judge Scoring Rubric

A detailed, unambiguous rubric is the constitution of the evaluation system. It is essential for aligning human labelers and for providing a strong, clear signal to the LLM-as-a-Judge during few-shot learning.¹⁰

| Criterion | Score 5 (Excellent) | Score 3 (Acceptable) | Score 1 (Poor) |
| :--- | :--- | :--- | :--- |
| **Hallucination** | Fully grounded. All facts/numbers are directly verifiable in the source context. | Mostly grounded, but may contain minor, non-critical inferences or rephrasing that slightly deviates. No factual errors. | Contains significant factual errors or fabricated information not present in the context. |
| **Captured Rate** | Captures all critical and most secondary points from the human reference. | Captures all critical points but may miss some secondary details or nuance. | Fails to capture one or more critical points from the human reference. |
| **Value Add** | Surfaces a non-obvious, highly relevant insight by synthesizing information from multiple chunks that provides a new perspective. | Surfaces a minor, useful fact or connection that was not in the human reference. | Provides no new information beyond what the human expert captured. |
| **Missing Out** | No critical or secondary information from the human reference is missing. | A few minor, non-essential details from the human reference are omitted. | Omits critical information that was present in the human reference, leading to an incomplete picture. |
| **Confidence** | Confidence tags are perfectly calibrated. Reports HIGH on fully grounded facts and LOW/MEDIUM on speculative or inferred points. | Confidence tags are mostly reasonable but may be slightly over/under-confident on some points. | Confidence is miscalibrated. Reports HIGH confidence on hallucinated facts or LOW confidence on simple, grounded facts. |

-----

## Section 3: Ensuring Cross-Model Consistency and Performance Alignment

A modern, production-grade AI strategy cannot rely on a single model provider. A multi-model approach is a strategic imperative for mitigating vendor lock-in, optimizing the cost-to-performance ratio for different sub-tasks, and leveraging the unique, specialized capabilities of a diverse range of models. The TinyRAG system is therefore designed to operate with a portfolio of leading commercial and open-source LLMs.

### 3.1 A Curated Portfolio for Financial Analysis

The models selected for the TinyRAG project are chosen for their specific strengths, which map to different roles within the RAG pipeline.

  * **OpenAI (GPT & o-series):** This family serves as the high-quality, reliable baseline for generation and evaluation. The GPT-4.1 series, including GPT-4.1-mini and GPT-4.1-nano, offers significant improvements in instruction following and long-context comprehension, with all models supporting a 1 million token context window.²¹ The `o-series`, particularly `o1-mini`, is specialized for complex reasoning tasks, making it an ideal candidate for the "heavy-lifter" role in automated regeneration loops where a difficult initial generation needs to be corrected [User Query].
  * **Google (Gemini):** The Gemini 2.5 Pro model is a frontier-level powerhouse, distinguished by its native multimodality and an exceptionally large 1 million token context window.²³ This makes it uniquely suited for tasks involving the deep analysis of financial reports that contain a mix of text, tables, and charts. While its capabilities are state-of-the-art, its deployment is subject to practical considerations, as access via Vertex AI can have geographical availability restrictions that must be factored into a global deployment strategy [User Query].
  * **Open-Source (Llama & Pixtral):** The open-source ecosystem provides strategic autonomy, data privacy, and granular control. Meta's Llama 4 Scout employs a highly efficient Mixture-of-Experts (MoE) architecture, activating only a fraction of its 109 billion parameters for any given task.²⁵ With an industry-leading 10 million token context window, it is purpose-built for deep, on-premise analysis of entire document repositories or codebases.²⁷ Mistral AI's `Pixtral-Large-Instruct-2411` is a natively multimodal model that has demonstrated state-of-the-art performance on document understanding benchmarks like DocVQA.²⁸ This makes it a prime candidate for the initial ingestion and intelligent chunking phase, especially for visually complex documents.

### 3.2 Model Provider Capability Matrix

To translate this portfolio into an actionable strategy, the following matrix maps the unique capabilities of each model family to specific roles within the TinyRAG system. This serves as a clear guide for the engineering team on how to allocate resources and select the right tool for the right job.

#### Table 3: Model Provider Capability Matrix

| Model Family | Key Model(s) | Architecture | Context Window | Key Strengths / Benchmarks | Strategic Role in TinyRAG |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenAI** | GPT-4.1-mini, o1-mini | Dense / Specialized | 1M Tokens | High general reasoning, instruction following, coding.²² | Baseline Generation, LLM-as-a-Judge, Low-Score Regeneration. |
| **Google** | Gemini 2.5 Pro | Dense, Multimodal | 1M Tokens | Frontier multimodal capabilities, long-context analysis.²³ | RAG over complex visual reports, alternative high-quality generator. |
| **Meta (Open)** | Llama 4 Scout | MoE (17B/109B) | 10M Tokens | Extreme long-context, efficient inference, on-premise control.²⁵ | On-premise fine-tuning, deep analysis of entire codebases/dockets. |
| **Mistral (Open)** | Pixtral-Large-Instruct | Multimodal Decoder | 128K Tokens | SOTA on DocVQA, chart understanding.²⁸ | Ingestion/Chunking of visual documents, multimodal Q\&A. |

### 3.3 The Consistency Challenge and Mitigation Strategies

A multi-model approach introduces a new challenge: ensuring that performance is aligned and evaluation is consistent across all models. The goal is not for all models to be identical, but for the evaluation framework to be fair and unbiased, allowing for meaningful comparisons. The primary threat to this consistency is the potential for bias in the LLM-as-a-Judge itself. Research has shown that LLMs can exhibit several forms of bias, including:

  * **Narcissistic Bias:** A tendency to favor outputs generated by models from the same family or with a similar style.¹⁶
  * **Position Bias:** A tendency to prefer the first item presented in a pairwise comparison, regardless of its quality.¹⁵
  * **Verbosity Bias:** A preference for longer, more verbose answers, even if they are less concise or accurate.¹⁶

If left unaddressed, these biases would render the evaluation metrics meaningless and undermine the goal of achieving aligned performance. The evaluation protocol must therefore be designed to be robust against these failure modes. The following strategies are essential for mitigating judge bias:

  * **Use a Neutral, Powerful Judge:** Whenever possible, the LLM-as-a-Judge should be a capable model from a different provider than the models being evaluated. For example, using a model from Anthropic (like Claude 3.5 Sonnet) or a fine-tuned open-source model to judge the outputs of OpenAI and Google models can help reduce self-preference bias.
  * **Pairwise Comparison with Swapping:** For head-to-head comparisons between two model outputs (e.g., Model A vs. Model B), the evaluation should be run twice. In the first run, the prompt presents Model A's output followed by Model B's. In the second run, the order is swapped. The final scores are then averaged, effectively canceling out any systematic position bias.
  * **Calibrate with Diverse Few-Shot Examples:** The few-shot examples provided in the judge's prompt are critical for calibration. These examples should intentionally include stylistically different but equally correct responses that all receive a high score (e.g., a score of 5). This teaches the judge to focus on the substantive quality of the answer (factuality, relevance, coherence) rather than superficial stylistic patterns.

By implementing these mitigation strategies, the evaluation framework can produce fair, reliable, and comparable metrics across the entire model portfolio, enabling a truly strategic and data-driven approach to model selection and deployment in the TinyRAG system.

---

**📚 Continue Your RAG Journey:**  
← **Previous:** [Retrieval Strategies in Financial RAG Systems: From Dense to Hybrid Approaches](/post/retrieval-strategies-in-financial-rag)  
→ **Next:** [Agentic Post-Processing in TinyRAG: Conditional, State-Driven Workflows](/post/agentic-post-processing-tinyrag)  
📋 **[View Complete Learning Path](/knowledge/rag)** | **Progress: 8/9 Complete** ✅✅✅✅✅✅✅✅
