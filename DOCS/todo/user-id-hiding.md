**Verdict:** the core pattern is sound and is the right one for this problem. The system as it stands isn't production-grade yet, because the trust boundaries around the pattern are weak.

## What it gets right

- **The model has no say in identity.** The parameter is hidden from the schema and then overwritten, so prompt injection and hallucination can't change whose data is read. Hiding alone relies on the model behaving. Overwriting alone leaves the model confused. Together they don't depend on the model at all.
- **It's enforced at a chokepoint.** The interceptor sits on every MCP call, so individual tools can't forget to apply it.
- **It's cheap and testable.** There are no extra LLM calls and no new infrastructure.

This is really an authorization control (preventing one customer reading another's data), not PII hiding. A customer ID like `cust003` is only a pseudonymous key. The sensitive data is what comes back from the tools, such as names, addresses and order details. Those tool results do reach the LLM, and so does anything the user types. This mechanism doesn't protect those, and your docs shouldn't imply it does.

## What I'd fix before production

1. **Authenticate the A2A hop (the biggest gap).** The remote agent trusts plain `metadata["customer_id"]` from whoever calls it. Anything that can reach the port can read any customer's data. Send a short-lived signed token (JWT) that carries the customer claim. The remote agent verifies it and derives `customer_id` from the verified claim, not from a free-text field. Use TLS in transit, plus network isolation, since agents shouldn't be publicly reachable.
2. **Fail closed.** `inject_customer_id` falls back to `DEFAULT_CUSTOMER_ID` when the config has none. In production a missing ID should raise an error. Today a bug in the config path would silently serve `cust001`'s data to someone else.
3. **Add defence in depth at the data layer.** Everything rests on one Python function. Add a second control, for example Postgres row-level security keyed on a session variable the MCP server sets, or a read-only DB role with parameterised views scoped per customer. Then a bug in the interceptor can't leak data.
4. **Bind threads to customers.** The caller also supplies `thread_id`, which the checkpointer uses as the state key. If a caller can guess another customer's thread ID, it could resume their conversation state. Namespace it, e.g. `{customer_id}:{uuid}`, and check on resume that the thread belongs to the verified customer.
5. **Guard against drift.** The hiding is by name convention (`customer_id`). Add a test that loads every tool and asserts none exposes `customer_id` in its model-facing schema, and that every tool call receives one. A new tool or a renamed parameter would otherwise quietly break the guarantee.
6. **Minimise what flows to the LLM.** Select only the columns the answer needs, and don't return the ID in results. The conversation history sent to the remote agent in the A2A data part can also contain PII, so decide whether the remote agent needs all of it.
7. **Check your observability.** I haven't verified whether Langfuse records the injected arguments or the MCP request body. Check a trace. The ID is also in logs and `langfuse_user_id` today. Hash or mask it if your policy treats it as personal data, in line with the `CLAUDE.md` logging rules.

## Summary

| Area | Status |
|---|---|
| Model can't see or set the ID | Strong |
| Authorization enforced in one place | Good |
| Caller authentication between agents | Missing |
| Fail-closed behaviour | Missing |
| Data-layer backstop (RLS or scoped role) | Missing |
| Thread-to-customer binding | Missing |
| Regression tests for the guarantee | Missing |
| PII in tool results, history and logs | Not covered |

Items 1 and 2 are the minimum before real customer traffic. I can implement the JWT check and the fail-closed change if you want, and update the doc's "known gaps" section to match this list.