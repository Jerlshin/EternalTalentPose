

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final

__all__: tuple[str, ...] = (
    "StageNode",
    "OfflineExecutionGraph",
    "OFFLINE_EXECUTION_GRAPH",
)


_STAGE_ID_RE: Final = re.compile(r"^O(\d+)([a-z]?)$")


def _stage_sort_key(stage_id: str) -> tuple[int, str]:
    """Return a natural-order key for a stage id.

    ``"O2"`` → ``(2, "")``; ``"O13a"`` → ``(13, "a")``. This makes tie-breaking
    numeric-then-suffix rather than lexicographic, so ``O2`` precedes ``O13a``.

    Raises:
        ValueError: if ``stage_id`` does not match the ``O<n><suffix>`` shape —
            a malformed graph definition (caught at construction).
    """
    match = _STAGE_ID_RE.match(stage_id)
    if match is None:
        msg = f"malformed stage id {stage_id!r} (expected 'O<n>' or 'O<n><letter>')"
        raise ValueError(msg)
    return (int(match.group(1)), match.group(2))


@dataclass(frozen=True, slots=True)
class StageNode:
    """One node of the DAG: a stage id, its dependencies, and criticality.

    Attributes:
        stage_id: The canonical stage id (``"O0"`` … ``"O18"``).
        depends_on: The stage ids that must complete before this one (its
            *upstream* edges, the "Deps" column of Part 2).
        critical: Whether a failure here aborts the whole build (fail-fast) or
            may be skipped under ``--continue-on-error`` (Part 11 failure
            recovery: "fail-fast (critical) or continue (non-critical, flagged)").
            Everything on the online-consumed artifact path is critical; the
            optional career-vector sub-stage (O13c) is the lone non-critical node.
    """

    stage_id: str
    depends_on: tuple[str, ...]
    critical: bool = True


# --------------------------------------------------------------------------- #
# The frozen DAG (Offline Pipeline Part 11). Edges are upstream dependencies.  #
#                                                                             #
#   O0 ─▶ O1 ─▶ O2 ─┬─▶ O3 ──────────────┐                                    #
#                   └─▶ O4 ─▶ O5 ─▶ O6 ──┤                                    #
#   O1 ─▶ O13a ─┬─▶ O7 ──────────────────┤                                    #
#   O6 ─▶ O13b ─┘                        │                                    #
#   {O4,O13a,O13b} ─▶ O14 ───────────────┼─▶ O8 ─▶ O9 ─▶ O10 ─┐               #
#                                        ├─▶ O11             │                #
#                                 {O3,O14}─▶ O12             │                #
#                          {O9,O10,O11,O12} ─▶ O15 ──────────┤                #
#                                 {O8,O10} ─▶ O16 ───────────┤                #
#                                                   all ─▶ O17 ─▶ O18         #
# --------------------------------------------------------------------------- #
_NODES: Final[tuple[StageNode, ...]] = (
    StageNode("O0", ()),
    StageNode("O1", ("O0",)),
    StageNode("O2", ("O1",)),
    StageNode("O3", ("O0", "O2")),
    StageNode("O4", ("O2",)),
    StageNode("O5", ("O4", "O13a")),
    StageNode("O6", ("O5",)),
    StageNode("O13a", ("O1",)),
    StageNode("O13b", ("O6",)),
    StageNode("O13c", ("O1",), critical=False),
    StageNode("O13", ("O13a", "O13b", "O13c")),
    StageNode("O7", ("O13a",)),
    StageNode("O14", ("O4", "O13a", "O13b")),
    StageNode("O8", ("O14", "O7")),
    StageNode("O9", ("O8", "O14")),
    StageNode("O10", ("O9", "O14")),
    StageNode("O11", ("O14",)),
    StageNode("O12", ("O3", "O14")),
    StageNode("O15", ("O9", "O10", "O11", "O12")),
    StageNode("O16", ("O8", "O10")),
    StageNode(
        "O17",
        (
            "O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O13", "O13a",
            "O13b", "O13c", "O8", "O9", "O10", "O11", "O12", "O14", "O15", "O16",
        ),
    ),
    StageNode("O18", ("O17",)),
)


@dataclass(frozen=True, slots=True)
class OfflineExecutionGraph:
    """Immutable O0–O18 dependency DAG with a deterministic plan.

    Construct via :data:`OFFLINE_EXECUTION_GRAPH` or :meth:`default`.
    Construction validates that every dependency references a known node and
    that the graph is acyclic (cycles caught at plan time, never at run time).
    """

    nodes: tuple[StageNode, ...]
    _by_id: Mapping[str, StageNode] = field(init=False, repr=False)
    _topo: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ids = [node.stage_id for node in self.nodes]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            msg = f"duplicate stage ids in execution graph: {', '.join(dupes)}"
            raise ValueError(msg)
        by_id = {node.stage_id: node for node in self.nodes}
        for node in self.nodes:
            for dep in node.depends_on:
                if dep not in by_id:
                    msg = (
                        f"stage {node.stage_id!r} depends on unknown stage {dep!r}"
                    )
                    raise ValueError(msg)
        object.__setattr__(self, "_by_id", by_id)
        object.__setattr__(self, "_topo", self._compute_topo(by_id))

    @classmethod
    def default(cls) -> OfflineExecutionGraph:
        """Return the graph built from the frozen Part 11 topology."""
        return cls(nodes=_NODES)

    @staticmethod
    def _compute_topo(by_id: Mapping[str, StageNode]) -> tuple[str, ...]:
        """Deterministic Kahn topological sort, ties broken by natural stage id.

        Raises:
            ValueError: if the graph contains a cycle — the remaining nodes are
                named so the config error is actionable (Part 1 failure modes:
                "DAG cycle (config error, caught at ``plan()``)").
        """
        indegree: dict[str, int] = {sid: 0 for sid in by_id}
        dependents: dict[str, list[str]] = {sid: [] for sid in by_id}
        for node in by_id.values():
            for dep in node.depends_on:
                indegree[node.stage_id] += 1
                dependents[dep].append(node.stage_id)

        # Ready set kept sorted by natural id so the order is fully deterministic.
        ready: list[str] = sorted(
            (sid for sid, deg in indegree.items() if deg == 0),
            key=_stage_sort_key,
        )
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            newly_ready: list[str] = []
            for dependent in dependents[current]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    newly_ready.append(dependent)
            if newly_ready:
                ready = sorted(ready + newly_ready, key=_stage_sort_key)

        if len(order) != len(by_id):
            remaining = sorted(set(by_id) - set(order), key=_stage_sort_key)
            msg = f"execution graph contains a cycle among: {', '.join(remaining)}"
            raise ValueError(msg)
        return tuple(order)

    def node(self, stage_id: str) -> StageNode:
        """Return the node for ``stage_id`` (``KeyError`` if unknown)."""
        try:
            return self._by_id[stage_id]
        except KeyError as exc:
            msg = f"unknown stage id {stage_id!r} in execution graph"
            raise KeyError(msg) from exc

    def stage_ids(self) -> tuple[str, ...]:
        """Return every stage id in deterministic topo order."""
        return self._topo

    def topo_order(self) -> tuple[str, ...]:
        """Return the deterministic topological execution order (alias)."""
        return self._topo

    def dependencies(self, stage_id: str) -> tuple[str, ...]:
        """Return the immediate upstream dependencies of ``stage_id``."""
        return self.node(stage_id).depends_on

    def dependents(self, stage_id: str) -> tuple[str, ...]:
        """Return the stage ids that directly depend on ``stage_id`` (topo-sorted)."""
        direct = [
            node.stage_id
            for node in self.nodes
            if stage_id in node.depends_on
        ]
        return tuple(sorted(direct, key=_stage_sort_key))

    def downstream_closure(self, stale: Iterable[str]) -> frozenset[str]:
        """Return ``stale`` plus every stage transitively downstream of it.

        This is the lineage-driven invalidation set (Part 11): if a stage's
        output changes, the runner must recompute the transitive closure of its
        dependents. The seed ids are included in the result.

        Raises:
            KeyError: if any seed id is unknown.
        """
        for sid in stale:
            self.node(sid)  # validate membership; raises on unknown id.
        closure: set[str] = set()
        frontier: list[str] = list(stale)
        while frontier:
            current = frontier.pop()
            if current in closure:
                continue
            closure.add(current)
            frontier.extend(self.dependents(current))
        return frozenset(closure)

    def parallel_layers(self) -> tuple[tuple[str, ...], ...]:
        """Return stages grouped into dependency layers for concurrent scheduling.

        Layer ``i`` contains every stage whose longest dependency chain has
        length ``i``; all stages in a layer are mutually independent and may run
        concurrently (Part 11: "the runner schedules independent branches
        concurrently; merges are deterministic"). Within a layer, ids are sorted
        naturally so any deterministic merge sees a fixed order.
        """
        depth: dict[str, int] = {}
        for stage_id in self._topo:
            deps = self.dependencies(stage_id)
            depth[stage_id] = 1 + max((depth[d] for d in deps), default=-1)
        max_depth = max(depth.values(), default=-1)
        layers: list[tuple[str, ...]] = []
        for level in range(max_depth + 1):
            members = sorted(
                (sid for sid, d in depth.items() if d == level),
                key=_stage_sort_key,
            )
            layers.append(tuple(members))
        return tuple(layers)


OFFLINE_EXECUTION_GRAPH: Final[OfflineExecutionGraph] = (
    OfflineExecutionGraph.default()
)