"""NTB-IR: the single representation of a model.

Two levels: the authored spatial level (this package) and NTB-Core IR, the flat
geometry-free DAG produced by ``ntb.spatial.resolve``. See docs/adr/0001, 0002.
"""

from ntb.ir.core import CoreEdge, CoreGraph, CoreNode
from ntb.ir.document import Document, Module
from ntb.ir.graph import Edge, Endpoint, Generator, Node, Port, PortDirection
from ntb.ir.spatial import Axis, Orientation, Placement, SpatialRule, SpatialRuleKind
from ntb.ir.types import DType, Shape, TensorType

__all__ = [
    "Axis",
    "CoreEdge",
    "CoreGraph",
    "CoreNode",
    "DType",
    "Document",
    "Edge",
    "Endpoint",
    "Generator",
    "Module",
    "Node",
    "Orientation",
    "Placement",
    "Port",
    "PortDirection",
    "Shape",
    "SpatialRule",
    "SpatialRuleKind",
    "TensorType",
]
