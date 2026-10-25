"""
Calculate retrieval evaluation metrics from a saved retrieval evaluation JSON file.

! Recall@K note:

Recall@K requires the complete set of relevant chunk IDs for each question.
Each result should include "relevant_chunk_ids" containing the ground-truth IDs.
If this field is absent or empty, recall is reported as None rather than
calculating a misleading value from only the chunks returned by the retriever.
"""

import json
import statistics
from collections import defaultdict
from ..config import PROJECT_ROOT

INPUT_PATH = (
    PROJECT_ROOT
    / "llm"
    / "data"
    / "test_dataset"
    / "retrieval_evaluation_results.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "llm"
    / "data"
    / "test_dataset"
    / "retrieval_metrics.json"
)

K = 5

def precision_at_k(chunks: list[dict], k: int = K) -> float:
    """Calculate Precision@K: relevant chunks divided by K."""

    # K must be positive; otherwise, the metric is undefined.
    if k <= 0:
        raise ValueError("k must be greater than 0")

    # Count how many of the top K chunks are marked as relevant.
    # Missing "is_relevant" fields are treated as False.
    relevant = sum(bool(chunk.get("is_relevant", False)) for chunk in chunks[:k])

    # Divide by K, not by the number of returned chunks.
    # Therefore, returning fewer than K chunks can lower precision.
    return relevant / k


def precision_returned(chunks: list[dict], k: int = K) -> float:
    """Calculate precision using the number of chunks actually returned."""

    if k <= 0:
        raise ValueError("k must be greater than 0")

    # Keep only the first K chunks.
    top_chunks = chunks[:k]

    # Avoid division by zero when the retriever returns no chunks.
    if not top_chunks:
        return 0.0

    # Count the relevant chunks among the results actually returned.
    relevant = sum(bool(chunk.get("is_relevant", False)) for chunk in top_chunks)

    # Unlike Precision@K, divide by the actual number of returned chunks.
    return relevant / len(top_chunks)


def recall_at_k(
    chunks: list[dict],
    relevant_chunk_ids: list[str] | set[str] | None,
    k: int = K,
) -> float | None:
    """Calculate Recall@K using ground-truth relevant chunk IDs."""

    if k <= 0:
        raise ValueError("k must be greater than 0")

    # Without ground-truth IDs, we cannot know the total number of relevant
    # chunks for this question. Return None instead of an unreliable score.
    if relevant_chunk_ids is None:
        return None

    # Convert the ground-truth IDs into a set.
    # Sets make it easy to find the intersection between two collections
    # and avoid counting duplicate IDs more than once.
    relevant_ids = {str(chunk_id) for chunk_id in relevant_chunk_ids}

    # An empty ground-truth set gives an undefined recall score because
    # the recall denominator would be zero.
    if not relevant_ids:
        return None

    # Collect the IDs of the top K retrieved chunks.
    # The JSON field is named "chunk_id", so use that exact key.
    # Ignore chunks that do not contain an ID.
    retrieved_ids = {
        str(chunk["chunk_id"])
        for chunk in chunks[:k]
        if chunk.get("chunk_id") is not None
    }

    # Find the chunks that appear in BOTH sets:
    # retrieved_ids & relevant_ids gives the relevant chunks successfully retrieved.
    # Divide their count by the total number of ground-truth relevant chunks.
    return len(retrieved_ids & relevant_ids) / len(relevant_ids)


def reciprocal_rank(chunks: list[dict], k: int = K) -> float:
    """Calculate the reciprocal rank of the first relevant chunk in the top K."""

    if k <= 0:
        raise ValueError("k must be greater than 0")

    # Enumerate the top K chunks with ranks starting from 1.
    for rank, chunk in enumerate(chunks[:k], start=1):

        # Stop at the first relevant chunk because MRR considers
        # the rank of the earliest relevant result.
        if chunk.get("is_relevant", False):

            # A relevant chunk at rank 1 gives 1.0;
            # rank 2 gives 0.5; rank 3 gives approximately 0.333.
            return 1 / rank

    # Return zero if none of the top K chunks is relevant.
    return 0.0


def calculate_metrics(data: dict) -> dict:
    """Calculate aggregate metrics separately for each retrieval method."""

    # Group results by retrieval method so similarity, threshold,
    # and hybrid retrieval are evaluated independently.
    grouped = defaultdict(list)

    for result in data["results"]:
        grouped[result["retrieval_method"]].append(result)

    # This dictionary will store the final aggregate metrics for each method.
    metrics = {}

    # Process each retrieval method in alphabetical order.
    for method, results in sorted(grouped.items()):

        # Store per-question metric values so we can calculate their averages.
        precisions = []
        returned_precisions = []
        recalls = []
        reciprocal_ranks = []
        times = []
        chunk_counts = []

        # Count questions for which the retriever returned no chunks.
        empty_results = 0

        # Calculate metrics for every question evaluated with this method.
        for result in results:

            # Sort retrieved chunks by rank to ensure that the first K
            # entries represent the highest-ranked results.
            # Use a large fallback rank if a chunk has no rank field.
            # Then retain only the top K chunks.
            chunks = sorted(
                result.get("retrieved_chunks", []),
                key=lambda chunk: chunk.get("rank", 999999),
            )[:K]

            # Calculate precision using K as the denominator.
            precisions.append(precision_at_k(chunks))

            # Calculate precision using the actual number of returned chunks.
            returned_precisions.append(precision_returned(chunks))

            # Calculate the reciprocal rank of the first relevant chunk.
            reciprocal_ranks.append(reciprocal_rank(chunks))

            # Record how many chunks were returned, up to K.
            chunk_counts.append(len(chunks))

            # This must contain the complete ground-truth relevant ID list
            # for the question, not just the relevant chunks retrieved.
            ground_truth_ids = result.get("relevant_chunk_ids")

            # Calculate recall for this question using the ground truth.
            recall = recall_at_k(chunks, ground_truth_ids)

            # None means recall could not be evaluated.
            # Do not include such questions in the recall average.
            if recall is not None:
                recalls.append(recall)

            # Track questions where the retriever returned no chunks.
            if not chunks:
                empty_results += 1

            # Record execution time only when it is numeric.
            execution_time = result.get("execution_time_ms")
            if isinstance(execution_time, (int, float)):
                times.append(execution_time)

        # Calculate the mean of each metric across the questions for this method.
        # Round the values to make the output easier to read.
        metrics[method] = {
            "question_count": len(results),

            # Average Precision@K across all questions.
            "precision_at_k": round(statistics.mean(precisions), 4),

            # Average precision among the chunks actually returned.
            "precision_among_returned": round(
                statistics.mean(returned_precisions), 4
            ),

            # Average recall across questions with available ground truth.
            # If no question has valid ground-truth IDs, report None.
            "recall_at_k": round(statistics.mean(recalls), 4) if recalls else None,

            # Number of questions that contributed to the recall average.
            # This can be lower than question_count if ground truth is missing.
            "recall_questions_evaluated": len(recalls),

            # Mean Reciprocal Rank across all questions.
            "mrr_at_k": round(statistics.mean(reciprocal_ranks), 4),

            # Average execution time in milliseconds, if available.
            "average_execution_time_ms": (
                round(statistics.mean(times), 2) if times else None
            ),

            # Average number of chunks returned per question.
            "average_chunks_returned": round(statistics.mean(chunk_counts), 2),

            # Total number of questions with no retrieved chunks.
            "empty_result_count": empty_results,
        }

    # Return the completed metrics dictionary.
    return metrics


def main() -> None:
    """Load retrieval results, calculate metrics, and save the report."""

    # Stop execution with a clear error if the input file does not exist.
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Results file not found: {INPUT_PATH}")

    # Open and parse the retrieval evaluation JSON file into a Python dictionary.
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Calculate aggregate metrics for every retrieval method.
    metrics = calculate_metrics(data)

    # Ensure the output directory exists before saving the metrics.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the metrics as formatted JSON for later inspection or comparison.
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Print the location of the saved metrics file.
    print(f"Metrics saved to: {OUTPUT_PATH}\n")

    # Print a table header for the metrics displayed in the terminal.
    print(
        f"{'Method':<12} {'P@5':>8} {'Recall@5':>10} {'MRR@5':>8} "
        f"{'Avg ms':>10} {'Avg chunks':>12} {'Empty':>8}"
    )
    print("-" * 82)

    # Display one summary row for each retrieval method.
    for method, values in metrics.items():

        # Format execution time for display.
        # Show N/A if no valid execution times were recorded.
        avg_time = values["average_execution_time_ms"]
        avg_time_text = f"{avg_time:.1f}" if avg_time is not None else "N/A"

        # Format recall for display.
        # N/A means recall could not be calculated, not that recall equals zero.
        recall = values["recall_at_k"]
        recall_text = f"{recall:.3f}" if recall is not None else "N/A"

        # Print the method's metrics in aligned columns.
        print(
            f"{method:<12} "
            f"{values['precision_at_k'] * 100:>7.1f}% "
            f"{recall_text:>10} "
            f"{values['mrr_at_k']:>8.3f} "
            f"{avg_time_text:>10} "
            f"{values['average_chunks_returned']:>12.2f} "
            f"{values['empty_result_count']:>8}"
        )

        # If recall is None, explain how to make it evaluable.
        if values["recall_at_k"] is None:
            print(
                " " * 4
                + "Recall@5 unavailable: add complete 'relevant_chunk_ids' "
                "ground-truth IDs to each question result."
            )


# Run main() only when this file is executed directly as a script.
# Importing this module from another file will not automatically run main().
if __name__ == "__main__":
    main()