"""Evaluation script: retrieval precision@k and answer faithfulness.

Run:  python -m src.evaluation.evaluator

Metrics:
  - Retrieval Precision@5: fraction of top-5 retrieved docs containing
    at least one relevant keyword for the question.
  - Answer Faithfulness: Gemini judges whether the answer is grounded in
    the retrieved context (scored 0–1 per question).

Results are printed to stdout and saved to data/eval_results.json.
"""

import json
import os
import time
from pathlib import Path

from google import genai
from dotenv import load_dotenv

from src.pipeline import rag_chain

load_dotenv()

QA_PATH = Path(__file__).parent / "qa_pairs.json"
RESULTS_PATH = Path("data/eval_results.json")

FAITHFULNESS_PROMPT = """You are evaluating whether an AI answer is faithful to its source documents.

Question: {question}

Retrieved context:
{context}

AI Answer:
{answer}

Score the answer's faithfulness on a scale of 0 to 1:
- 1.0: Every claim in the answer is directly supported by the context
- 0.75: Most claims are supported; minor extrapolation
- 0.5: About half the claims are supported; some unsupported assertions
- 0.25: Few claims are grounded; significant hallucination
- 0.0: Answer contradicts or ignores the context entirely

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<one sentence>"}}"""


def retrieval_precision_at_k(retrieved_docs: list, keywords: list[str], k: int = 5) -> float:
    """Fraction of top-k docs containing at least one relevant keyword."""
    if not retrieved_docs or not keywords:
        return 0.0
    top_k = retrieved_docs[:k]
    hits = sum(
        1
        for doc in top_k
        if any(kw.lower() in doc.page_content.lower() or kw.lower() in str(doc.metadata).lower()
               for kw in keywords)
    )
    return hits / len(top_k)


def answer_faithfulness(
    question: str,
    answer: str,
    context_docs: list,
    client: genai.Client,
) -> tuple[float, str]:
    """Use Gemini Flash to score whether the answer is grounded in the context."""
    context_text = "\n\n".join(doc.page_content[:400] for doc in context_docs)
    prompt = FAITHFULNESS_PROMPT.format(
        question=question,
        context=context_text,
        answer=answer,
    )
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        return float(result["score"]), result.get("reason", "")
    except Exception as e:
        return 0.5, f"Eval error: {e}"


def run_eval(k: int = 5) -> dict:
    with open(QA_PATH) as f:
        qa_pairs = json.load(f)

    # Use Flash for eval — fast and free
    judge_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    results = []

    print(f"\nRunning evaluation on {len(qa_pairs)} questions (k={k})\n{'='*60}")

    total_precision = 0.0
    total_faithfulness = 0.0

    for qa in qa_pairs:
        qid = qa["id"]
        question = qa["question"]
        keywords = qa["relevant_keywords"]
        country_filter = qa.get("relevant_countries")
        doc_type_filter = qa.get("relevant_doc_types")

        print(f"\n[{qid}] {question}")

        result = rag_chain.query(
            question=question,
            country_filter=country_filter,
            doc_type_filter=doc_type_filter,
            n_results=k,
        )

        precision = retrieval_precision_at_k(result.sources, keywords, k=k)
        total_precision += precision
        print(f"  Retrieval Precision@{k}: {precision:.2f} ({result.n_docs_retrieved} docs retrieved)")

        if result.sources:
            faith_score, faith_reason = answer_faithfulness(
                question, result.answer, result.sources, judge_client
            )
            time.sleep(0.5)  # Rate limit courtesy
        else:
            faith_score, faith_reason = 0.0, "No documents retrieved"

        total_faithfulness += faith_score
        print(f"  Faithfulness:           {faith_score:.2f} — {faith_reason}")

        results.append(
            {
                "id": qid,
                "question": question,
                "precision_at_k": precision,
                "faithfulness": faith_score,
                "faithfulness_reason": faith_reason,
                "n_docs_retrieved": result.n_docs_retrieved,
                "answer_snippet": result.answer[:200],
            }
        )

    n = len(qa_pairs)
    summary = {
        "n_questions": n,
        "k": k,
        f"mean_precision_at_{k}": round(total_precision / n, 3),
        "mean_faithfulness": round(total_faithfulness / n, 3),
        "per_question": results,
    }

    print(f"\n{'='*60}")
    print(f"Mean Precision@{k}:  {summary[f'mean_precision_at_{k}']:.3f}")
    print(f"Mean Faithfulness:  {summary['mean_faithfulness']:.3f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {RESULTS_PATH}")

    return summary


if __name__ == "__main__":
    run_eval()
