from dotenv import load_dotenv
load_dotenv()

import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm.src.generation.prompt import SYSTEM_PROMPT
from llm.src.tools.search_tool import answer, fast_answer

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_retries=3,
)

prompt = ChatPromptTemplate.from_messages([
    ( "system", SYSTEM_PROMPT),
    ( "human",
        """Use the following retrieved context to answer the question.
        Retrieved context: {context}
        User question: {question}

        Answer using the system instructions.""",
    ),
])

# Convert the model response into a plain string
parser = StrOutputParser()

generation_runnable = prompt | llm | parser

def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into readable context for the LLM."""

    if not chunks:
        return "No relevant document chunks were retrieved."

    formatted_chunks = []
    for index, chunk in enumerate(chunks, start=1):
        formatted_chunks.append(
            f"""Chunk {index}
                Chunk ID: {chunk.get("id")}
                Metadata: {json.dumps(chunk.get("metadata", {}), ensure_ascii=False)}
                Content:
                {chunk.get("text", "")}"""
            )
        
    return "\n\n".join(formatted_chunks)


def generate_answer(question: str, mode: str = "answer") -> str:
    """
    Retrieve relevant chunks and generate an answer.

    Modes:
        answer: Dense similarity search.
        fast_answer: Hybrid search.
    """

    # Select the retrieval tool based on the requested mode.
    if mode == "answer":
        chunks = answer.invoke({"query": question})

    elif mode == "fast_answer":
        chunks = fast_answer.invoke({"query": question})

    else:
        raise ValueError(
            "Invalid mode. Choose 'answer' or 'fast_answer'."
        )

    # Format the retrieved chunks for the prompt.
    context = format_context(chunks)

    # Generate the answer using the context and question.
    response = generation_runnable.invoke(
        {
            "context": context,
            "question": question,
        }
    )
    return response


if __name__ == "__main__":

    question = input("Kindly enter your question:\n").strip()
    if not question or not question.strip():
        raise ValueError("Kindly enter your question.")
    
    print("\nChoose retrieval mode:")
    print("1. Answer")
    print("2. Faster Answer")

    mode_choice = input("Enter 1 or 2: ").strip()

    if mode_choice == "1":
        mode = "answer"
    elif mode_choice == "2":
        mode = "fast_answer"
    else:
        print("Invalid choice. Please enter 1 or 2.")
        raise SystemExit(1)

    if question:
        result = generate_answer(question, mode=mode)
        print("\nAnswer:\n")
        print(result)
