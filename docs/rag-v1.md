# Quizy RAG V1

This version changes the curriculum pipeline from "retrieve a few chunks and answer" to a gated retrieval pipeline designed for educational material.

## Request flow

```text
student message
  -> intent classification
  -> follow-up rewrite
  -> document routing
  -> high-recall dense + lexical + exact retrieval
  -> semantic reranker
  -> retrieval quality gate
       -> accept
       -> retry query once
       -> abstain
  -> parent/child context expansion
  -> structured grounded answer
  -> clean student-facing answer + structured sources
```

## Contextual hierarchical chunks

Newly indexed documents use smaller child chunks than the previous 900-token default. The effective child limit is controlled with:

```env
CONTEXTUAL_CHUNK_MAX_TOKENS=550
CONTEXTUAL_CHUNK_OVERLAP_TOKENS=80
```

Each child gets an internal context prefix containing the inferred chapter, lesson, section, page and content type. A page-level `parent_context` chunk is also indexed. Initial retrieval excludes parent chunks; a parent is loaded only after a child on that page is selected.

This gives us the `index small, retrieve big` pattern without a database migration.

## Semantic reranking

Dense, lexical and exact matching are now candidate-generation signals. The configured chat model reranks the best candidates according to whether each passage can actually support the student's exact question.

```env
SEMANTIC_RERANKER_ENABLED=true
SEMANTIC_RERANKER_CANDIDATE_K=14
SEMANTIC_RERANKER_TOP_K=6
SEMANTIC_RERANKER_MAX_CHARS_PER_CANDIDATE=1200
SEMANTIC_RERANKER_WEIGHT=0.78
```

If the reranker returns malformed structured output, retrieval falls back safely to the hybrid score instead of failing the request.

## Retrieval gate and retry

The gate prevents an academic answer just because some vaguely related chunk exists.

```env
RETRIEVAL_GATE_ACCEPT_SCORE=0.60
RETRIEVAL_GATE_RETRY_SCORE=0.32
RETRIEVAL_GATE_MIN_HYBRID_SCORE=0.16
RETRIEVAL_GATE_MAX_RETRIES=1
```

The states are:

- `accept`: evidence is strong enough to answer.
- `retry`: evidence is plausible but weak; the service rewrites the search query once and retrieves again.
- `abstain`: evidence is still insufficient; Quizy tells the student the information is not in their current curriculum.

`/api/chat/ask` and the final stream event expose `retrievalDiagnostics` so evaluation can track gate status, score and retries. This is diagnostic metadata; it is not part of the student-facing answer.

## Structured answer generation

Curriculum answers are generated internally as:

```json
{
  "answer": "...",
  "usedSourceIds": ["S1"],
  "confidence": 0.82,
  "groundingMode": "direct"
}
```

The API returns `answer` as normal text and maps `usedSourceIds` to the existing `sources` array. Internal `S1`/`S2` IDs are explicitly forbidden in the visible answer and are stripped defensively as a final guard.

The public Quizy backend integration does not need to change.

## Deployment

No SQL migration is required for V1 because parent context uses the existing `document_chunks` table and `content_type` column.

However, **existing PDFs need one retry/reindex after deployment**. Old chunks continue to work, but they do not receive contextual hierarchy or parent chunks until they are rebuilt.

Recommended rollout:

1. Deploy the new RAG service.
2. Keep the existing PDFs and document IDs.
3. Trigger the existing `POST /api/documents/{documentId}/retry` endpoint for each ready curriculum PDF.
4. Wait until each document returns to `ready`.
5. Run the V1 evaluation dataset against staging before production rollout.

## Evaluation

Use:

```bash
python scripts/evaluate_rag.py evals/dataset.v1.example.json
```

Replace placeholder document IDs and enable curriculum cases first. The evaluator now checks:

- expected answer terms
- expected source pages
- answer vs abstention behavior
- routing status
- minimum grounded confidence
- direct vs derived grounding mode
- retrieval retry count
- source-marker leakage
- estimated cost

It also reports aggregate average confidence, average retries, routing counts and source-marker leaks.

CI runs `scripts/check_rag_v1.py` to protect hierarchy generation, parent chunks, gate behavior, source-marker cleanup and route replacement.
