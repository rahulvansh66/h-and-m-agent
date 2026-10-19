# Customer support RAG: remaining work

The customer support agent is a `condense_query -> rag -> respond` subgraph. The relevance
gate (`rag.min_score`) and the delimited-document prompt are done. The items below are the
production hardening that is still open, roughly in the order to do them.

## 1. Scope retrieval to the right documents
- The Qdrant collection `troubleshooting_kb` also holds other articles (for example undeliverable
  packages). Add a payload filter on `source_file` (or `doc_title`) so this agent answers only
  from `damaged-or-defective-item.pdf`.
- Put the filter value in `config/base.yaml` under `rag`, not in code.
- Create a payload index on the filtered field in Qdrant.

## 2. Return an outcome, and hand off when there is no answer
- Make the RAG result carry `outcome: answered | no_answer` instead of only text.
- On `no_answer`, hand off to a human or open a ticket instead of the fixed "Sorry, I could not
  find..." message. For a damaged item a dead end is the worst result for the customer.
- Open question: for now the reply could be a fixed message pointing to customer service; decide
  whether to create a real ticket.

## 3. Check that a citation actually supports the claim
- Today a claim passes if it cites a real retrieved document number. It is not checked that the
  cited text supports the claim.
- Start with a lexical-overlap check between the claim and the cited chunk (no extra LLM call).
  Consider a small-model verification step only if overlap proves too weak (costs one extra Groq
  call per query).

## 4. Verify supervisor routing for follow-up turns (implemented, not yet run)
- The supervisor now sees the last `supervisor.history_messages` turns, labelled with the agent
  that wrote each reply, and has an `out_of_scope` route. This was meant to fix "How long does
  the refund for it take?" going to `order_management`.
- Run it against Groq once: a damaged-item turn followed by that refund follow-up should stay
  with `customer_support`, and an unrelated message should get the out-of-scope reply.
- Agent labels depend on the `name` set on each agent's AI messages; check that the order
  agent's replies carry `name="order_management"`.

## 5. Async and resilience
- Use `ainvoke` end to end and run the fastembed call in a thread (it is CPU-bound and would
  block the event loop).
- Add timeouts and a single retry on Qdrant calls.
- If Groq rate-limits, fall back to `GROQ_FALLBACK_API_KEY` (already listed in `.env.example`).
- If the knowledge base is unreachable, return the human-handoff reply instead of an error.

## 6. Cold start and deployment
- fastembed downloads about 130 MB on first use. Bake the model into the container image so it is
  not fetched at runtime.
- If this is deployed to Lambda, cold starts will be noticeable. Hosted embeddings are the
  alternative, but they mean a different embedding model and re-embedding the whole collection,
  so avoid that unless cold start proves to be a problem.

## 7. Observability
- The Langfuse callback already reaches the nested subgraphs. Add the `thread_id` to the RAG log
  lines so one conversation can be traced across nodes.
- Keep logging counts and scores only, never the customer's text.

## 8. Tests
- Unit tests with the LLM and retriever mocked: cited answer; uncited, then retry, then cited;
  uncited twice (no-answer); all chunks below `min_score` (no answer-generation call).
- Supervisor: damaged-item message, order message, off-topic message (fixed out-of-scope reply,
  retrieval never called), follow-up that stays with the previous agent, and a structured-output
  failure (falls back to `customer_support`).
- `condense_query`: no LLM call on the first turn, one call on a follow-up.
- A small golden set of 5 to 10 questions run occasionally against the real stack (Groq is
  rate-limited, so keep it small and manual).
- Re-tune `rag.min_score` (currently 0.5) against that golden set. Observed scores so far: about
  0.42 for an off-topic query, 0.67 to 0.77 for real questions.

## 9. Out of scope for now
- Damaged-item claims often depend on order details (what was bought, when). Agents stay
  independent per the project rules, so chaining the order and support agents is left to the
  supervisor later.
