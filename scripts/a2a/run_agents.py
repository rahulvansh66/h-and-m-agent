"""Start the A2A agent services and keep them running until Ctrl+C.

Run (from the repo root):
    uv run python scripts/a2a/run_agents.py
    uv run python scripts/a2a/run_agents.py --log-dir DOCS/working-info/a2a

Each agent is a separate process (``python -m hm_agent.<agent>.a2a_service``) on the port set in
``config/base.yaml``. With ``--log-dir`` each process's console output goes to ``agent-<name>.log``
there instead of this terminal. The launcher waits until every Agent Card answers, then blocks.
"""

import argparse
import logging
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from hm_agent.core.logging_setup import configure_logging
from hm_agent.core.settings import REPO_ROOT, agent_url
from hm_agent.core.state import CUSTOMER_SUPPORT, ORDER_MANAGEMENT, SHIPMENT_TRACKING

logger = logging.getLogger(__name__)

AGENTS = (ORDER_MANAGEMENT, CUSTOMER_SUPPORT, SHIPMENT_TRACKING)
CARD_PATH = "/.well-known/agent-card.json"
STARTUP_TIMEOUT_S = 120  # the support agent loads a local embedding model, which can take a while


def _wait_for_card(agent: str, proc: subprocess.Popen) -> bool:
    """Poll an agent's Agent Card URL until it answers or the process dies.

    Args:
        agent (str): Agent name.
        proc (subprocess.Popen): The agent's process; polling stops if it exits.

    Returns:
        bool: True when the card was served within ``STARTUP_TIMEOUT_S``, otherwise False.
    """
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline and proc.poll() is None:
        try:
            with urllib.request.urlopen(agent_url(agent) + CARD_PATH, timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def main() -> None:
    """Launch all agent services, wait for them to be ready, and stop them on Ctrl+C."""
    configure_logging("a2a-launcher")
    parser = argparse.ArgumentParser(description="Run the H&M A2A agent services")
    parser.add_argument("--log-dir", default=None, help="Directory for per-agent console logs (default: this terminal)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    procs: dict[str, subprocess.Popen] = {}
    try:
        for agent in AGENTS:
            out = open(log_dir / f"agent-{agent}.log", "w", encoding="utf-8") if log_dir else None
            procs[agent] = subprocess.Popen(
                [sys.executable, "-m", f"hm_agent.{agent}.a2a_service"],
                cwd=REPO_ROOT,
                stdout=out,
                stderr=subprocess.STDOUT if out else None,
            )
            logger.info("agent=%s status=started pid=%s url=%s", agent, procs[agent].pid, agent_url(agent))
        for agent, proc in procs.items():
            ready = _wait_for_card(agent, proc)
            logger.info("agent=%s status=%s", agent, "ready" if ready else "not_ready")
            if not ready:
                raise SystemExit(f"agent {agent} did not become ready; check its log")
        print("All agents ready. Press Ctrl+C to stop.", flush=True)
        while all(p.poll() is None for p in procs.values()):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for agent, proc in procs.items():
            proc.terminate()
        for agent, proc in procs.items():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            logger.info("agent=%s status=stopped", agent)


if __name__ == "__main__":
    main()
