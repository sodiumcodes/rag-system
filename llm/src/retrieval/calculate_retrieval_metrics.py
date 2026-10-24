"""
Calculate retrieval evaluation metrics from a saved retrieval evaluation JSON file.

This module:
1. Reads retrieval_evaluation_results.json and, 
calculates precision and ranking metrics for each retrieval method.

2. Saves aggregate and per-question metrics to retrieval_metrics.json.
"""

import json
import statistics
from collections import defaultdict
from ..config import PROJECT_ROOT

INPUT_PATH = (
    PROJECT_ROOT / "llm" / "data" / "test_dataset"
    / "retrieval_evaluation_results.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "llm" / "data" / "test_dataset"
    / "retrieval_metrics.json"
)

K = 5
# Measures the proportion of the top 5 retrieved chunks that are relevant, using k as the denominator.
def precision_at_k(chunks: list[dict], k: int = K) -> float:
    """Relevant chunks in top K divided by K."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant = sum(chunk.get("is_relevant", False) for chunk in chunks[:k])
    return relevant / k

# Measures how many of the chunks actually returned are relevant,
def precision_returned(chunks: list[dict], k: int = K) -> float:
    """Relevant chunks divided by the number actually returned."""
    chunks = chunks[:k]

    if not chunks:
        return 0.0

    relevant = sum(chunk.get("is_relevant", False) for chunk in chunks)
    return relevant / len(chunks)

# Measures how highly the first relevant chunk is ranked across questions.
def reciprocal_rank(chunks: list[dict], k: int = K) -> float:
    """Reciprocal rank of the first relevant chunk."""
    for rank, chunk in enumerate(chunks[:k], start=1):
        if chunk.get("is_relevant", False):
            return 1 / rank
    return 0.0

def calculate_metrics(data: dict) -> dict:
    """Calculate metrics for each retrieval method."""

    grouped = defaultdict(list) #!creates a dictionary where each new key automatically gets an empty list as its default value.
    
    for result in data["results"]:
        grouped[result["retrieval_method"]].append(result)
    metrics = {}

    for method, results in sorted(grouped.items()):
        precisions = []
        returned_precisions = []
        reciprocal_ranks = []
        times = []
        chunk_counts = []
        empty_results = 0

        for result in results:
            chunks = sorted(
                result.get("retrieved_chunks", []),
                key=lambda chunk: chunk.get("rank", 999999),
            )[:K]

            precisions.append(precision_at_k(chunks))
            returned_precisions.append(precision_returned(chunks))
            reciprocal_ranks.append(reciprocal_rank(chunks))
            chunk_counts.append(len(chunks))

            if not chunks:
                empty_results += 1

            execution_time = result.get("execution_time_ms")
            if isinstance(execution_time, (int, float)):
                times.append(execution_time)

        metrics[method] = {
            "question_count": len(results),
            "precision_at_k": round(statistics.mean(precisions), 4),
            "precision_among_returned": round(
                statistics.mean(returned_precisions), 4
            ),
            "mrr_at_k": round(statistics.mean(reciprocal_ranks), 4),
            "average_execution_time_ms": (
                round(statistics.mean(times), 2) if times else None
            ),
            "average_chunks_returned": round(
                statistics.mean(chunk_counts), 2
            ),
            "empty_result_count": empty_results,
        }

    return metrics

def main() -> None:
    """Load results, calculate metrics, and save the report."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Results file not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    metrics = calculate_metrics(data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"Metrics saved to: {OUTPUT_PATH}\n")
    print(
        f"{'Method':<12} {'P@5':>8} {'MRR@5':>8} "
        f"{'Avg ms':>10} {'Avg chunks':>12} {'Empty':>8}"
    )
    print("-" * 62)

    for method, values in metrics.items():
        avg_time = values["average_execution_time_ms"]
        avg_time = f"{avg_time:.1f}" if avg_time is not None else "N/A"

        print(
            f"{method:<12} "
            f"{values['precision_at_k'] * 100:>7.1f}% "
            f"{values['mrr_at_k']:>8.3f} "
            f"{avg_time:>10} "
            f"{values['average_chunks_returned']:>12.2f} "
            f"{values['empty_result_count']:>8}"
        )


if __name__ == "__main__":
    main()