#!/bin/sh
# Start the A2A agent named by $AGENT (order_management, customer_support, shipment_tracking, fashion).
#
# The fashion agent loads its tools from a long-lived MCP server while it builds, so that server is started
# first (in this same container) and awaited. Its host/port come from config/base.yaml, same as locally.
set -e

if [ "$AGENT" = "fashion" ]; then
  python -m hm_agent.fashion.mcp_server &
  until python -c "import socket; from hm_agent.core.settings import FASHION_MCP_HOST as h, FASHION_MCP_PORT as p; socket.create_connection((h, p), 2)" 2>/dev/null; do
    sleep 1
  done
fi

exec python -m "hm_agent.${AGENT}.a2a_service"
