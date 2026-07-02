from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chain_manager import ChainManager
    from .node_proxy import NodeProxy
    from .chain_proxy import ChainProxy

from . import node_proxy as _node_proxy
from . import chain_proxy as _chain_proxy


class ComponentProxy:
    """Represents one connected component."""

    def __init__(self, chain: ChainManager, comp_id: int) -> None:
        self._chain: ChainManager = chain
        self._id: int = comp_id

    @property
    def id(self) -> int:
        return self._id

    @property
    def nodes(self) -> list[NodeProxy]:
        return [
            _node_proxy.NodeProxy(self._chain, n)
            for n in self._chain.get_component_members(self._id)
        ]

    @property
    def size(self) -> int:
        return len(self._chain.get_component_members(self._id))

    def get_chains(self, minimal_required: bool = True) -> list[ChainProxy]:
        all_chains: list[list[str]] = list(self._chain.get_chains().values())
        comp_nodes: set[str] = set(self._chain.get_component_members(self._id))
        matching: list[tuple[int, list[str]]] = []
        i: int
        c: list[str]
        for i, c in enumerate(all_chains):
            if any(n in comp_nodes for n in c):
                matching.append((i, c))
        return [_chain_proxy.ChainProxy(self._chain, i, c) for i, c in matching]

    def __repr__(self) -> str:
        return f"ComponentProxy(id={self._id}, size={self.size})"
