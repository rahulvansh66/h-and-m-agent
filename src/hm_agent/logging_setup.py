"""Logging setup shared by every entry point (CLI, RAG CLI, MCP servers).

Role in the system: the one place that configures the root logger. Console output is always
on; when ``logging.cloudwatch.enabled`` is true in the config, records are also shipped to an
AWS CloudWatch Logs group (created by ``infra/cloudwatch``). Library modules never call this;
they only use ``logging.getLogger(__name__)``.
"""

import logging
import os
import subprocess

import boto3
import watchtower

from hm_agent.config import REPO_ROOT, load_config

logger = logging.getLogger(__name__)

# key=value fields (service/env/commit added by _ContextFilter) so CloudWatch Logs Insights can query them.
LOG_FORMAT = "%(levelname)s service=%(service)s env=%(env)s commit=%(commit)s logger=%(name)s %(message)s"


def _git_commit() -> str:
    """Return the short git commit hash of the checkout, or ``unknown`` if unavailable.

    Returns:
        str: Short commit hash, or ``unknown`` when git is missing or this is not a checkout.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class _ContextFilter(logging.Filter):
    """Stamp every record with the service name, environment and git commit."""

    def __init__(self, service: str, env: str, commit: str) -> None:
        """Store the correlation fields.

        Args:
            service (str): Entry point name, e.g. ``cli`` or ``mcp-order-management``.
            env (str): Active ``APP_ENV``.
            commit (str): Short git commit hash.
        """
        super().__init__()
        self._fields = {"service": service, "env": env, "commit": commit}

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach the fields to the record.

        Args:
            record (logging.LogRecord): Record being emitted.

        Returns:
            bool: Always True; the record is never dropped.
        """
        for key, value in self._fields.items():
            setattr(record, key, value)
        return True


def _cloudwatch_handler(cfg: dict, service: str) -> logging.Handler | None:
    """Build the CloudWatch Logs handler, or None if AWS is unreachable or misconfigured.

    Uses an explicit named profile (``CLOUDWATCH_AWS_PROFILE``) so it never picks up the
    ``AWS_ACCESS_KEY_ID`` in ``.env``, which belongs to Neon's S3-compatible storage.

    Args:
        cfg (dict): The ``logging.cloudwatch`` config section.
        service (str): Entry point name; one log stream per service per day.

    Returns:
        logging.Handler | None: A batching handler, or None when it could not be created.
    """
    try:
        session = boto3.Session(profile_name=os.environ.get("CLOUDWATCH_AWS_PROFILE"), region_name=cfg["region"])
        return watchtower.CloudWatchLogHandler(
            boto3_client=session.client("logs"),
            log_group_name=cfg["log_group"],
            log_stream_name=service + "/{strftime:%Y-%m-%d}",
            create_log_group=False,  # created by Terraform with retention set
            send_interval=10,
        )
    except Exception:  # logging must never stop the app from starting
        logger.exception("component=cloudwatch status=disabled reason=handler_init_failed")
        return None


def configure_logging(service: str) -> None:
    """Configure the root logger for an entry point (console, plus CloudWatch if enabled).

    Call once from the entry point after ``load_dotenv``.

    Args:
        service (str): Entry point name, used as a field on every record and as the
            CloudWatch stream prefix, e.g. ``cli``.
    """
    config = load_config()
    env = config["environment"]
    context = _ContextFilter(service, env, _git_commit())

    handlers: list[logging.Handler] = [logging.StreamHandler()]  # stderr: MCP stdio uses stdout
    cw_cfg = config["logging"]["cloudwatch"]
    cw_handler = _cloudwatch_handler(cw_cfg, service) if cw_cfg["enabled"] else None
    if cw_handler:
        handlers.append(cw_handler)

    formatter = logging.Formatter(LOG_FORMAT)
    for handler in handlers:
        handler.addFilter(context)
        handler.setFormatter(formatter)
    logging.basicConfig(level=config["logging"]["level"], handlers=handlers, force=True)
    if cw_handler:
        logger.info("component=cloudwatch status=enabled log_group=%s", cw_cfg["log_group"])
