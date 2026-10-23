# Commands used to produce the logs in this folder

Run from the repo root. Terminal 1 starts the two agent services, terminal 2 runs the chat.

```bash
# Terminal 1: start the 3 A2A agent services; each one's console output goes to agent-<name>.log
uv run python scripts/a2a/run_agents.py --log-dir DOCS/working-info/a2a        # launcher output -> launcher.log

# Terminal 2: save the Agent Cards (-> agent-cards.txt)
curl -s http://127.0.0.1:8001/.well-known/agent-card.json | python -m json.tool
curl -s http://127.0.0.1:8002/.well-known/agent-card.json | python -m json.tool

# Terminal 2: one chat session, two turns on one thread (-> session-console.log, a2a-wire.jsonl)
export A2A_WIRE_LOG=DOCS/working-info/a2a/a2a-wire.jsonl
printf 'What was the last item I bought?\nThe zipper on my jacket arrived broken, what can I do?\nexit\n' \
  | uv run python -m hm_agent --customer-id cust001 > DOCS/working-info/a2a/session-console.log 2>&1
```

`langfuse-observations.txt` was produced afterwards from the Langfuse v2 observations API
(`GET /api/public/v2/observations`, filtered by `sessionId`).

| File | What it is |
|---|---|
| `session-console.log` | What the chat terminal printed: supervisor and A2A client logs plus the answers. |
| `a2a-wire.jsonl` | Every HTTP request/response between supervisor and agents, one JSON object per line (8 lines). |
| `agent-order_management.log`, `agent-customer_support.log` | Console output of each agent service (uvicorn access lines and our logs). |
| `launcher.log` | Launcher start/ready lines. |
| `agent-cards.txt` | The Agent Cards as served (captured before `out_of_scope` became a local node, so it still lists a third card). |
| `langfuse-observations.txt` | Langfuse traces and spans for this session. |
| `agent-down-test.log` | One query with all agents stopped (`uv run python -m hm_agent "Tell me a joke"`): shows the failure path. |
