"""A2A (Agent2Agent) layer: each agent runs as its own HTTP service, the supervisor is the client.

Role in the system: ``server`` wraps a graph as an A2A server, ``client`` calls one, ``cards``
describes each agent (Agent Card), ``remote_node`` plugs a remote agent into the supervisor graph,
and each agent package has an ``a2a_service`` module as its runnable entry point.
"""
