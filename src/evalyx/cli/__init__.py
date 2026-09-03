"""Evalyx CLI/TUI: a REST client for the Evalyx evaluation platform.

The CLI never touches PostgreSQL, Redis, Celery, or application endpoints
directly — it talks only to the Evalyx API, which stays authoritative for
authentication, authorization, evaluation, and secrets.
"""

__version__ = "0.1.0"
