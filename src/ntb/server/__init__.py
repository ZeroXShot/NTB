"""FastAPI + WebSocket studio server.

Only :mod:`ntb.server.app` needs the `server` extra; sessions and the op catalog
are plain Python, so the CLI and the tests can use them without FastAPI.
"""

from ntb.server.catalog import describe_op, op_catalog
from ntb.server.session import Derived, Session, blank_document

__all__ = ["Derived", "Session", "blank_document", "describe_op", "op_catalog"]
