"""Scan-owned provenance graph model.

Roadmap goal: every ProbeFlow operation that produces a new image
(transformation) emits an :class:`ImageNode`; every operation that produces
a measurement (atom count, lattice vector, line profile) emits a
:class:`MeasurementNode`. Each node records its inputs (``parent_ids``),
the operation that produced it, the parameters, the plugin/version that
ran it, any warnings, and the result's units.

ImageNode arrays are virtual by default: ``array`` is ``None`` and the
recipe (``parent_ids`` + ``operation`` + ``params``) is sufficient to
reproduce the array on demand. Call :func:`materialize_image` to compute
the array by walking the recipe; call :meth:`ImageNode.release` on a
derived node to drop the cached array and reclaim memory. Roots — the
raw scan channels — always carry their array; they are the immutable
starting points of every chain.

Boundary rules
--------------
This module owns *only* the dataclasses and graph operations. It must
not import GUI code, vendor parsers, or numerical kernels. Operation
implementations are looked up via an ``op_registry`` callable supplied
by the caller (the plugin registry, the in-tree processing module, or
a test mock). That keeps the graph reusable across the CLI, the GUI,
and headless batch scripts without dragging in PySide6 or scipy.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Union

import numpy as np


OpRegistry = Callable[[str], Callable[..., np.ndarray]]


# ── Node types ────────────────────────────────────────────────────────────────


@dataclass
class ImageNode:
    """One image vertex in the scan provenance graph.

    Either a *root* (raw channel; ``parent_ids == ()``, ``operation == "root"``,
    ``array`` set) or *derived* (transformation output; the recipe is the
    parent_ids + operation + params, and ``array`` is ``None`` until
    materialized).
    """

    source_scan_id: str
    operation: str
    parent_ids: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    plugin_version: str = ""
    warnings: tuple[str, ...] = ()
    units: str = ""
    array: np.ndarray | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def is_root(self) -> bool:
        return self.operation == "root" and not self.parent_ids

    @property
    def is_virtual(self) -> bool:
        return self.array is None

    def release(self) -> None:
        """Drop the cached array on a derived node. Roots cannot be released."""
        if self.is_root:
            raise ValueError("Cannot release a root node's array")
        self.array = None


@dataclass
class MeasurementNode:
    """One measurement vertex in the scan provenance graph.

    Measurements project image data onto a result (a number, dict, array, or
    path to a saved artifact). Unlike ImageNodes, results are stored eagerly:
    they are typically small and re-running a measurement is rarely cheaper
    than caching the value.
    """

    source_scan_id: str
    operation: str
    parent_ids: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    plugin_version: str = ""
    warnings: tuple[str, ...] = ()
    units: str = ""
    result: Any = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


Node = Union[ImageNode, MeasurementNode]


# ── Graph container ───────────────────────────────────────────────────────────


@dataclass
class ScanGraph:
    """Provenance graph for a single scan.

    Owns a dict of nodes keyed by node id. Roots are the raw channels of
    the scan (e.g. Z-fwd, Z-bwd, I-fwd, I-bwd); derived nodes hang off
    them via ``parent_ids``. Add nodes via :meth:`add`; query topology
    with :meth:`parents` / :meth:`children`.
    """

    scan_id: str
    nodes: dict[str, Node] = field(default_factory=dict)
    root_ids: tuple[str, ...] = ()

    def add(self, node: Node, *, root: bool = False) -> str:
        """Insert ``node`` and return its id.

        Raises ``ValueError`` on id collision, dangling parent reference, or
        invalid root (a root must be an :class:`ImageNode` carrying its array
        and having no parents).
        """
        if node.id in self.nodes:
            raise ValueError(f"Node id {node.id!r} already in graph")
        for parent_id in node.parent_ids:
            if parent_id not in self.nodes:
                raise ValueError(
                    f"Parent {parent_id!r} not in graph (node {node.id!r})"
                )
        if root:
            if not isinstance(node, ImageNode):
                raise ValueError("Root nodes must be ImageNodes")
            if node.parent_ids:
                raise ValueError("Root nodes cannot have parents")
            if node.array is None:
                raise ValueError("Root nodes must carry their array")
            self.root_ids = (*self.root_ids, node.id)
        self.nodes[node.id] = node
        return node.id

    def get(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def parents(self, node_id: str) -> tuple[Node, ...]:
        return tuple(self.nodes[pid] for pid in self.nodes[node_id].parent_ids)

    def children(self, node_id: str) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes.values() if node_id in n.parent_ids)

    def image_nodes(self) -> tuple[ImageNode, ...]:
        return tuple(n for n in self.nodes.values() if isinstance(n, ImageNode))

    def measurement_nodes(self) -> tuple[MeasurementNode, ...]:
        return tuple(
            n for n in self.nodes.values() if isinstance(n, MeasurementNode)
        )

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes.values())

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.nodes


# ── Materialization ───────────────────────────────────────────────────────────


def materialize_image(
    graph: ScanGraph,
    node_id: str,
    op_registry: OpRegistry,
) -> np.ndarray:
    """Compute (and cache) an :class:`ImageNode`'s array by walking its recipe.

    Recursive: ensures all parent ImageNodes are materialized first. Returns
    the cached ``array`` directly when the node is already materialized
    (including all root nodes).

    Parameters
    ----------
    graph
        Scan graph holding the node and its ancestors.
    node_id
        Id of the ImageNode to materialize.
    op_registry
        Callable mapping an operation name to its implementation. The
        implementation receives parent arrays positionally (one per
        ``parent_id``) followed by ``**node.params``. The graph does not
        own this registry; pass the in-tree dispatcher, the plugin
        registry, or a test mock.
    """
    node = graph.get(node_id)
    if not isinstance(node, ImageNode):
        raise TypeError(
            f"materialize_image: node {node_id!r} is a {type(node).__name__}, "
            "expected ImageNode"
        )
    if node.array is not None:
        return node.array
    if node.is_root:
        raise RuntimeError(
            f"Root node {node_id!r} has no array attached; cannot materialize"
        )
    parent_arrays = tuple(
        materialize_image(graph, pid, op_registry) for pid in node.parent_ids
    )
    op_fn = op_registry(node.operation)
    if len(parent_arrays) == 1:
        node.array = op_fn(parent_arrays[0], **node.params)
    else:
        node.array = op_fn(*parent_arrays, **node.params)
    return node.array


# ── JSON serialization ────────────────────────────────────────────────────────


def _to_jsonable(result: Any) -> Any:
    """Convert a graph value to a JSON-friendly value.

    Numpy arrays become lists; tuples become lists; non-serializable objects
    fall back to ``repr()`` so the round-trip is at least readable, but
    callers should prefer to store paths to artifacts rather than raw
    objects whenever possible.
    """
    if result is None or isinstance(result, (str, int, float, bool)):
        return result
    if isinstance(result, np.generic):
        return result.item()
    if isinstance(result, (list, tuple)):
        return [_to_jsonable(x) for x in result]
    if isinstance(result, dict):
        return {str(k): _to_jsonable(v) for k, v in result.items()}
    if isinstance(result, np.ndarray):
        return result.tolist()
    return repr(result)


def _node_to_dict(node: Node) -> dict[str, Any]:
    base = {
        "id": node.id,
        "kind": "image" if isinstance(node, ImageNode) else "measurement",
        "source_scan_id": node.source_scan_id,
        "operation": node.operation,
        "parent_ids": list(node.parent_ids),
        "params": _to_jsonable(node.params),
        "plugin_version": node.plugin_version,
        "warnings": list(node.warnings),
        "units": node.units,
    }
    if isinstance(node, MeasurementNode):
        base["result"] = _to_jsonable(node.result)
    return base


def graph_to_dict(graph: ScanGraph) -> dict[str, Any]:
    """Serialize the graph to a JSON-friendly dict.

    Image arrays are *not* serialized — the recipe is sufficient to recompute
    them given the raw scan and the op registry. Roots, which carry arrays
    eagerly, must be re-armed by the caller after deserialization.
    """
    return {
        "scan_id": graph.scan_id,
        "root_ids": list(graph.root_ids),
        "nodes": [_node_to_dict(n) for n in graph.nodes.values()],
    }


def graph_from_dict(payload: dict[str, Any]) -> ScanGraph:
    """Inverse of :func:`graph_to_dict`.

    Roots arrive without their array — the caller must call
    ``graph.get(root_id).array = raw_plane`` before any
    :func:`materialize_image` call that depends on the root.
    """
    graph = ScanGraph(scan_id=payload["scan_id"])
    for raw in payload.get("nodes", []):
        kind = raw.get("kind")
        if kind == "image":
            node: Node = ImageNode(
                id=raw["id"],
                source_scan_id=raw["source_scan_id"],
                operation=raw["operation"],
                parent_ids=tuple(raw.get("parent_ids", ())),
                params=dict(raw.get("params", {})),
                plugin_version=raw.get("plugin_version", ""),
                warnings=tuple(raw.get("warnings", ())),
                units=raw.get("units", ""),
                array=None,
            )
        elif kind == "measurement":
            node = MeasurementNode(
                id=raw["id"],
                source_scan_id=raw["source_scan_id"],
                operation=raw["operation"],
                parent_ids=tuple(raw.get("parent_ids", ())),
                params=dict(raw.get("params", {})),
                plugin_version=raw.get("plugin_version", ""),
                warnings=tuple(raw.get("warnings", ())),
                units=raw.get("units", ""),
                result=raw.get("result"),
            )
        else:
            raise ValueError(f"Unknown node kind {kind!r}")
        graph.nodes[node.id] = node
    graph.root_ids = tuple(payload.get("root_ids", ()))
    return graph


__all__ = [
    "ImageNode",
    "MeasurementNode",
    "Node",
    "OpRegistry",
    "ScanGraph",
    "graph_from_dict",
    "graph_to_dict",
    "materialize_image",
]
