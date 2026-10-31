# Images for the supervisor and the four A2A agents (linux/arm64, the architecture AgentCore Runtime runs).
#
# One multi-stage file:
#   base        Python + uv + locked dependencies + code (shared, cached by every other stage)
#   agent       an A2A agent service; pick it with --build-arg AGENT=<name>
#   supervisor  the AgentCore entrypoint (POST /invocations, GET /ping on :8080)
#
# Build (see docker-compose.yml, which does this for all five):
#   docker build --platform linux/arm64 --target agent --build-arg AGENT=order_management -t hm-order-management .
#
# No secrets live in the images: .env is excluded (.dockerignore) and values are passed at run time.

# ---- base ---------------------------------------------------------------------------------------
FROM --platform=linux/arm64 python:3.11-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

# The code finds config/ relative to src/, so it runs from /app/src (not installed into site-packages).
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# Dependencies first: this layer is rebuilt only when the lock file changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY config ./config
COPY src ./src

# Run as an unprivileged user.
RUN useradd --system --uid 1001 --no-create-home app
USER app

# ---- agent --------------------------------------------------------------------------------------
FROM base AS agent

ARG AGENT
# Agents that embed text locally (customer_support, fashion) bake the model into the image so a cold start
# does not download it. Set to "false" for the others to keep them small.
ARG PRELOAD_EMBEDDING=false
ENV AGENT=${AGENT} \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    # AgentCore Runtime requires 0.0.0.0:9000 for A2A agents.
    A2A_BIND_HOST=0.0.0.0 \
    A2A_BIND_PORT=9000

USER root
COPY docker/start-agent.sh /usr/local/bin/start-agent
RUN sed -i 's/\r$//' /usr/local/bin/start-agent && chmod +x /usr/local/bin/start-agent \
    && mkdir -p /opt/fastembed && chown app /opt/fastembed
USER app
RUN if [ "$PRELOAD_EMBEDDING" = "true" ]; then \
      python -c "import os; from hm_agent.core.config import load_config; from fastembed import TextEmbedding; TextEmbedding(load_config()[{'customer_support': 'rag', 'fashion': 'fashion'}[os.environ['AGENT']]]['embedding'])"; \
    fi

EXPOSE 9000
CMD ["start-agent"]

# ---- supervisor ---------------------------------------------------------------------------------
FROM base AS supervisor

EXPOSE 8080
CMD ["python", "-m", "hm_agent.runtime.entrypoint"]
