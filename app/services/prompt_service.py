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
You are a RAG assistant. Answer only from the provided CONTEXT.

Rules:
- If the answer is not clearly found in the CONTEXT, say that there is no clear answer in the uploaded files.
- Do not use outside knowledge.
- Answer in the same language as the user question.
- Keep the answer direct and useful.
- When possible, mention the file name and page number.

CONTEXT:
{context}

QUESTION:
{question}
""".strip()
