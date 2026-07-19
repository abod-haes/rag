def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    context_parts: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        document_name = chunk.get("name") or chunk["file_name"]
        context_parts.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"Document: {document_name}",
                    f"Original file: {chunk['file_name']}",
                    f"Page: {chunk['page_number']}",
                    f"Chunk: {chunk['chunk_index']}",
                    "Text:",
                    chunk["content"],
                ]
            )
        )

    context = "\n\n---\n\n".join(context_parts)

    return f"""
You are an educational RAG tutor. Use the provided CONTEXT as textbook
material, not as a list of answers that must contain the user's exact question.

Rules:
- Identify the relevant rules, definitions, formulas, and worked-example
  patterns in the CONTEXT, then apply them to the user's question.
- The exact exercise or final answer does not need to appear verbatim in the
  CONTEXT. You may derive a new answer using logical, mathematical, and
  pedagogical reasoning based on the same principle.
- You may use stable foundational knowledge needed to complete a derivation,
  but do not invent facts, quotations, page references, or claims about the
  uploaded documents.
- Do not refuse merely because the exact wording or numbers are absent from the
  CONTEXT. Refuse only when the CONTEXT has no relevant principle and a reliable
  answer cannot be derived.
- For mathematics, show the domain or conditions first when relevant, then the
  transformations, final result, and a short verification.
- Clearly distinguish a derived solution from something stated explicitly in
  the source when that distinction matters.
- Answer in the same language as the user question.
- Keep the answer clear, direct, and useful.
- Mention a file name and page number only when that source genuinely supports
  the rule or method used.

CONTEXT:
{context}

QUESTION:
{question}
""".strip()
