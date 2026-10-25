import json
import time
from llm.src.config import PROJECT_ROOT
from llm.src.retrieval.retriever import DocumentRetriever
from llm.src.tools.search_tool import get_retriever

DATASET_PATH = (
    PROJECT_ROOT
    / "llm"
    / "data"
    / "test_dataset"
    / "top_relevant_chunks_from_both_files.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "llm"
    / "data"
    / "test_dataset"
    / "retrieval_evaluation_results.json"
)

TOP_K = 5
FETCH_K = 20
MAX_DISTANCE = 0.7
RRF_K = 60

def load_dataset() -> list[dict]:
    """Load questions and validate their ground-truth relevant chunk IDs."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {DATASET_PATH}")

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    questions = data.get("questions", [])
    if not questions:
        raise ValueError("The evaluation dataset has no questions.")

    seen_question_ids = set()

    # Validate every question in the dataset before starting retrieval.
    for item in questions:

        if "question" not in item:
            raise ValueError(f"Question text missing: {item}")

        question_id = item.get("question_id", item.get("id"))

        if question_id is None:
            raise ValueError(f"Question ID missing: {item}")

        # Ensure that no two questions share the same ID.
        # Duplicate IDs could make evaluation results ambiguous.
        if question_id in seen_question_ids:
            raise ValueError(f"Duplicate question ID in dataset: {question_id}")

        # Record this ID so it can be checked against subsequent questions.
        seen_question_ids.add(question_id)

        # Check that the question includes its ground-truth relevant chunk IDs.
        # These IDs are needed to determine which retrieved chunks are relevant
        # and to calculate Recall@K.
        if "relevant_chunk_ids" not in item:
            raise ValueError(
                f"Ground-truth chunk IDs missing for question {question_id}"
            )

        # Retrieve the list of ground-truth relevant chunk IDs.
        relevant_ids = item["relevant_chunk_ids"]

        # The field must be a non-empty list.
        # An empty list would not provide a valid denominator for recall.
        if not isinstance(relevant_ids, list) or not relevant_ids:
            raise ValueError(
                f"'relevant_chunk_ids' must be a non-empty list for "
                f"question {question_id}"
            )

        # Check that every chunk ID is a non-empty string.
        # This prevents missing, blank, or incorrectly typed IDs.
        if any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in relevant_ids
        ):
            raise ValueError(
                f"Every relevant chunk ID must be a non-empty string "
                f"for question {question_id}"
            )

        # Ensure that the same relevant chunk ID is not listed more than once.
        # Duplicate IDs would incorrectly inflate the ground-truth count.
        if len(relevant_ids) != len(set(relevant_ids)):
            raise ValueError(
                f"Duplicate relevant chunk IDs for question {question_id}"
            )
        
    return questions

def evaluate_method(
    retriever: DocumentRetriever,
    question: str,
    question_id: int | str,
    method_name: str,
    relevant_chunk_ids: set[str],
) -> dict:
    """Run one retrieval method and record results plus ground truth."""

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
    execution_time_ms = (time.perf_counter() - start_time) * 1000

    retrieved_chunks = []

    for rank, result in enumerate(results, start=1):
        # The retriever returns the ID under "id"; keep the output key explicit.
        chunk_id = result.get("id")
        if chunk_id is not None:
            chunk_id = str(chunk_id)

        retrieved_chunks.append(
            {
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
            }
        )

    return {
        "question_id": question_id,
        "retrieval_method": method_name,
        "relevant_chunk_ids": sorted(relevant_chunk_ids),
        "retrieved_chunks": retrieved_chunks,
        "execution_time_ms": round(execution_time_ms, 3),
    }


def main() -> None:
    questions = load_dataset()
    retriever = get_retriever()

    methods = ["similarity", "threshold", "hybrid"]
    evaluation_results = []

    try:
        for question_item in questions:
            question_id = question_item.get(
                "question_id",
                question_item.get("id"),
            )
            question = question_item["question"]
            relevant_chunk_ids = set(question_item["relevant_chunk_ids"])

            print(f"\nQuestion {question_id}: {question}")

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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    print("\nEvaluation completed.")
    print(f"Questions: {len(questions)}")
    print(f"Retrieval runs: {len(evaluation_results)}")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
