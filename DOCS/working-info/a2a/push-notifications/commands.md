# Commands used to produce the logs in this folder

Run from the repo root. Terminal 1 starts the three agent services, terminal 2 runs the chat.
Everything runs locally (agents on 127.0.0.1:8001-8003, push receiver on 127.0.0.1:8010).

```bash
# Terminal 1: start the A2A agents; each agent's console output goes to agent-<name>.log
uv run python scripts/a2a/run_agents.py --log-dir DOCS/working-info/a2a/push-notifications \
  > DOCS/working-info/a2a/push-notifications/launcher.log 2>&1

# Terminal 2: save the Agent Cards of the blocking agent and the push agent (-> agent-cards.txt)
for p in 8001 8003; do curl -s http://127.0.0.1:$p/.well-known/agent-card.json | python -m json.tool; done \
  > DOCS/working-info/a2a/push-notifications/agent-cards.txt

# Terminal 2: one chat session on one thread (-> session-console.log, a2a-wire.jsonl)
#   turn 1 is asked at once; turn 2 is asked 20 s later, while the carrier task is still running
export A2A_WIRE_LOG=DOCS/working-info/a2a/push-notifications/a2a-wire.jsonl
(echo 'Where is my order ORD-1007?'; sleep 20; echo 'How many purchases have I made?'; sleep 25; echo exit) \
  | uv run python -m hm_agent --customer-id cust001 \
  > DOCS/working-info/a2a/push-notifications/session-console.log 2>&1
```

`thread-state.txt` was produced afterwards by reading the thread back from the Neon checkpointer
(`build_graph(checkpointer).get_state({"configurable": {"thread_id": "<thread id>"}})`).

To try it interactively instead: start the agents as above, then `uv run python -m hm_agent` and type
`Where is my order ORD-1007?`. The acknowledgement appears at once; the carrier answer is printed
about 20 s later as `assistant (shipment_tracking, update)>`.

| File | What it is |
|---|---|
| `session-console.log` | What the chat terminal printed: supervisor, A2A client and push receiver logs plus the answers. |
| `a2a-wire.jsonl` | Every HTTP body exchanged, one JSON object per line (16 lines). `C->S` / `S->C` are supervisor calls and responses; `PUSH A->S` are the notifications the agent POSTs to the supervisor's webhook. |
| `agent-shipment_tracking.log` | The push-enabled agent: task start, the tool call, and each "Push-notification sent" line. |
| `agent-order_management.log`, `agent-customer_support.log` | The other two agents (not used by the push flow; the order agent answers turn 2). |
| `agent-cards.txt` | Agent Cards served by `order_management` (push off) and `shipment_tracking` (push on). |
| `launcher.log` | Process start and ready lines. |
| `thread-state.txt` | The messages saved in the conversation thread after the session. |
