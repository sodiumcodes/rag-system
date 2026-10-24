import json
import time
from pathlib import Path

from llm.src.config import PROJECT_ROOT
from llm.src.retrieval.retriever import DocumentRetriever

# path to reviewed dataset containing relevant_chunk_ids.
DATASET_PATH = (
    PROJECT_ROOT
    / "llm"
    / "data"
    / "test_dataset"
    / "chunk_review.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "llm"
    / "data"
    / "test_dataset"
    / "retrieval_evaluation_results.json"
)

# RETRIEVAL SETTINGS
TOP_K = 5
FETCH_K = 20
MAX_DISTANCE = 0.7
RRF_K = 60

def load_dataset() -> list[dict]:
    """Load questions and their reviewed relevant chunk IDs."""

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    questions = data["questions"]

    if not questions:
        raise ValueError("The evaluation dataset has no questions.")

    for item in questions:
        if "question" not in item:
            raise ValueError(
                f"Question text missing: {item}"
            )

        if "relevant_chunk_ids" not in item:
            raise ValueError(
                f"Ground-truth chunk IDs missing for "
                f"question {item.get('question_id')}"
            )

    return questions

def evaluate_method(
    retriever: DocumentRetriever,
    question: str,
    question_id: int | str,
    method_name: str,
    relevant_chunk_ids: set[str],
) -> dict:
    """Run one retrieval method and record its results."""

    # Select the existing retrieval method.
    if method_name == "similarity":
        search_function = lambda: retriever.similarity_search(
            question, top_k=TOP_K
        )

    elif method_name == "threshold":
        search_function = lambda: retriever.threshold_search(
            question,
            top_k=TOP_K,
            max_distance=MAX_DISTANCE,
        )

    elif method_name == "hybrid":
        search_function = lambda: retriever.hybrid_search(
            question,
            top_k=TOP_K,
            fetch_k=FETCH_K,
            rrf_k=RRF_K,
        )

    else:
        raise ValueError(f"Unknown retrieval method: {method_name}")

    # Measure retrieval execution time only.
    start_time = time.perf_counter()
    results = search_function()
    execution_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    retrieved_chunks = []

    for rank, result in enumerate(results, start=1):
        chunk_id = result.get("id")

        retrieved_chunks.append({
            "chunk_id": chunk_id,
            "rank": rank,
            "distance": (
                float(result["distance"])
                if result.get("distance") is not None
                else None
            ),
            "rrf_score": (
                float(result["rrf_score"])
                if result.get("rrf_score") is not None
                else None
            ),
            "is_relevant": (
                chunk_id in relevant_chunk_ids
                if chunk_id is not None
                else False
            ),
        })

    return {
        "question_id": question_id,
        "retrieval_method": method_name,
        "retrieved_chunks": retrieved_chunks,
        "execution_time_ms": round(execution_time_ms, 3),
    }

def main():
    questions = load_dataset()
    retriever = DocumentRetriever()

    methods = [
        "similarity",
        "threshold",
        "hybrid",
    ]

    evaluation_results = []

    try:
        for question_item in questions:
            question_id = question_item.get(
                "question_id",
                question_item.get("id"),
            )

            if question_id is None:
                raise ValueError(
                    "Each question must have question_id or id."
                )

            question = question_item["question"]

            relevant_chunk_ids = set(
                question_item["relevant_chunk_ids"]
            )

            print(
                f"\nQuestion {question_id}: {question}"
            )

            for method_name in methods:
                result = evaluate_method(
                    retriever=retriever,
                    question=question,
                    question_id=question_id,
                    method_name=method_name,
                    relevant_chunk_ids=relevant_chunk_ids,
                )

                evaluation_results.append(result)

                print(
                    f"  {method_name}: "
                    f"{len(result['retrieved_chunks'])} chunks, "
                    f"{result['execution_time_ms']} ms"
                )

    finally:
        # Close the persistent Chroma client.
        retriever.client.close()

    output = {
        "total_questions": len(questions),
        "retrieval_methods": methods,
        "total_runs": len(evaluation_results),
        "results": evaluation_results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    print(f"\nEvaluation completed.")
    print(f"Questions: {len(questions)}")
    print(f"Retrieval runs: {len(evaluation_results)}")
    print(f"Results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
