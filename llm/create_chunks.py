
"""
Create a manual-review file for labeling relevant ChromaDB chunk IDs.

Run from the project root:
    python -m llm.create_chunks

Optional:
    python -m llm.create_chunks --top-k 10
    python -m llm.create_chunks --output llm/data/test_dataset/chunk_review.json

This script:
- Reads question_answer.json without modifying it.
- Retrieves candidate chunks for each question.
- Saves the candidates to a separate JSON review file.
- Does NOT automatically label any chunk as relevant.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from llm.src.tools.search_tool import get_retriever

def json_safe(value: Any) -> Any:
    """Convert common non-JSON values into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    # Handle NumPy scalars and similar objects.
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (ValueError, TypeError):
            pass

    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a JSON file for manually labeling relevant chunks."
    )

    parser.add_argument(
        "--dataset",
        default="llm/data/test_dataset/question_answer.json",
        help="Path to the existing QA JSON file, relative to project root.",
    )

    parser.add_argument(
        "--output",
        default="llm/data/test_dataset/chunk_review.json",
        help="Path for the separate review JSON file.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of candidate chunks per question (default: 5).",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    # Resolve paths relative to the project root.
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = (project_root / args.dataset).resolve()
    output_path = (project_root / args.output).resolve()

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Check the --dataset path."
        )

    # Allow imports such as llm.src.retrieval.retriever.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, dict):
        raise ValueError("The dataset root must be a JSON object.")

    questions = dataset.get("questions")

    if not isinstance(questions, list):
        raise ValueError(
            "Expected the dataset to contain a 'questions' list."
        )

    if not questions:
        raise ValueError("The dataset contains no questions.")

    print(f"Dataset: {dataset_path}")
    print(f"Questions found: {len(questions)}")
    print(f"Candidates per question: up to {args.top_k}\n")

    retriever = get_retriever()
    review_questions = []
    skipped = 0

    for position, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            print(f"[{position}/{len(questions)}] Skipped: invalid entry.")
            skipped += 1
            continue

        question = item.get("question", "")
        answer = item.get("answer", "")

        if not isinstance(question, str) or not question.strip():
            print(f"[{position}/{len(questions)}] Skipped: empty question.")
            skipped += 1
            continue

        # retrieved = retriever.similarity_search(
        #     query=question,
        #     top_k=args.top_k,
        # )
        retrieved = retriever.hybrid_search(
            query=question,
            top_k=args.top_k,
        )

        candidate_chunks = []

        for rank, chunk in enumerate(retrieved, start=1):
            candidate_chunks.append(
                {
                    "rank": rank,
                    "id": chunk.get("id"),
                    "text": chunk.get("text", ""),
                    "metadata": json_safe(
                        chunk.get("metadata", chunk.get("metadatas", {}))
                    ),
                    "distance": json_safe(chunk.get("distance")),
                    "is_relevant": None,
                }
            )

        review_questions.append(
            {
                "question_id": position,
                "category": item.get("category"),
                "question": question,
                "answer": answer,
                "relevant_chunk_ids": [],
                "candidate_chunks": candidate_chunks,
                "review_instruction": (
                    "Inspect each candidate's text. Set is_relevant to true "
                    "or false for each candidate. Add the IDs of all relevant "
                    "chunks to relevant_chunk_ids. Retrieval alone does not "
                    "mean a chunk is relevant. If no candidate is relevant, "
                    "search the collection for missing evidence."
                ),
            }
        )

        print(
            f"[{position}/{len(questions)}] "
            f"Retrieved {len(candidate_chunks)} candidates: "
            f"{question[:11]}"
        )

    review_data = {
        "dataset_name": dataset.get(
            "dataset_name", "QA evaluation dataset"
        ),
        "source_document": dataset.get("source_document"),
        "purpose": (
            "Manual review of candidate ChromaDB chunks for retrieval "
            "evaluation. Candidates are not automatically labeled."
        ),
        "top_k_candidates_per_question": args.top_k,
        "original_question_count": len(questions),
        "processed_question_count": len(review_questions),
        "skipped_question_count": skipped,
        "questions": review_questions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(review_data, file, ensure_ascii=False, indent=2)

    print("\nReview file created successfully.")
    print(f"Output: {output_path}")
    print(f"Questions processed: {len(review_questions)}")
    print(f"Questions skipped: {skipped}")
    print(
        "\nNext: manually review the candidate chunks, set "
        "'is_relevant' to true or false, and fill in "
        "'relevant_chunk_ids'."
    )


if __name__ == "__main__":
    main()