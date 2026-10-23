# Commands used to produce the logs in this folder

Run from the repo root. Terminal 1 starts the three agent services, terminal 2 runs the chat.
Everything runs locally (agents on 127.0.0.1:8001-8003, push receiver on 127.0.0.1:8010, inside the chat process).

```bash
# Terminal 1: start the A2A agents; each agent's console output goes to agent-<name>.log
uv run python scripts/a2a/run_agents.py --log-dir DOCS/working-info/a2a/streaming \
  > DOCS/working-info/a2a/streaming/launcher.log 2>&1

# Terminal 2: save the Agent Cards of the streaming-only agent and the streaming + push agent (-> agent-cards.txt)
for p in 8001 8003; do curl -s http://127.0.0.1:$p/.well-known/agent-card.json | python -m json.tool; done \
  > DOCS/working-info/a2a/streaming/agent-cards.txt

# Terminal 2: one chat session, one thread, three questions (-> session-console.log, a2a-wire.jsonl)
export A2A_WIRE_LOG=DOCS/working-info/a2a/streaming/a2a-wire.jsonl
(echo 'What was the last item I bought?'; sleep 15; \
 echo 'Was that one bought online or in a store?'; sleep 15; \
 echo 'Where is my order ORD-1007?'; sleep 40; echo exit) \
  | uv run python -m hm_agent --customer-id cust001 \
  > DOCS/working-info/a2a/streaming/session-console.log 2>&1
```

`thread-state.txt` was produced afterwards by reading the thread back from the Neon checkpointer
(`build_graph(checkpointer).get_state({"configurable": {"thread_id": "<thread id>"}})`).
The `token` value of the webhook registration in `a2a-wire.jsonl` was replaced by `<redacted>`
(it is a random per-process secret).

To try it interactively: start the agents as above, then `uv run python -m hm_agent`. Progress lines are printed as
`  ... text`, streamed reply text appears as it arrives.

| File | What it is |
|---|---|
| `session-console.log` | What the chat terminal printed: supervisor, A2A client and push receiver logs, progress lines and the answers. |
| `a2a-wire.jsonl` | Every HTTP body exchanged, one JSON object per line (35 lines). `C->S` is a supervisor request, `S->C` a plain response, `S->C SSE` one Server-Sent-Event frame of a streaming response, `PUSH A->S` a notification the agent POSTs to the supervisor's webhook. |
| `agent-order_management.log` | The order agent: task start / complete (with number of chunks sent) per question. |
| `agent-shipment_tracking.log` | The streaming + push agent: tool calls, `Push-notification sent` lines and one failed notification (see the explainer, section 7). Its last task (context `28193115-...`) is a separate check run after the receiver fix, not part of the session. |
| `agent-customer_support.log` | Third agent (not used by this session). |
| `agent-cards.txt` | Agent Cards of `order_management` (streaming) and `shipment_tracking` (streaming + push). |
| `launcher.log` | Process start and ready lines. |
| `thread-state.txt` | The messages saved in the conversation thread after the session. |
| `plan.md` | The plan this work followed. |
