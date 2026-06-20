from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .chain_manager import ChainManager
    from .chain_proxy import ChainProxy
    from .component_proxy import ComponentProxy


class NodeProxy:
    """Represents one image/node in the graph. Created on demand, zero overhead."""

    def __init__(self, chain: ChainManager, node_id: str, image_data: dict[str, Any] | None = None) -> None:
        self._chain: ChainManager = chain
        self._node_id: str = node_id
        self._image_data: dict[str, Any] = image_data or {}

    @property
    def id(self) -> str:
        return self._node_id

    @property
    def filename(self) -> str:
        return self._node_id

    @property
    def score(self) -> float:
        return self._image_data.get("score", 0.5)

    @property
    def mu(self) -> float:
        return self._image_data.get("rating_mu", 25.0)

    @property
    def sigma(self) -> float:
        return self._image_data.get("rating_sigma", 25.0 / 3.0)

    @property
    def comparison_count(self) -> int:
        return self._image_data.get("comparison_count", 0)

    @property
    def chain_count(self) -> int:
        """Number of chains this node belongs to. Should match comparison_count."""
        return len(self._chain.get_node_chains(self._node_id))

    @property
    def main_chain_in_chains(self) -> bool:
        """True if the main chain is in the list of all chains."""
        main: tuple[int, list[str]] | None = self._chain.get_node_main_chain(self._node_id)
        if main is None:
            return False
        all_chains: list[tuple[int, list[str]]] = self._chain.get_node_chains(self._node_id)
        return any(c[0] == main[0] for c in all_chains)

    @property
    def prompt_tags(self) -> str | None:
        return self._image_data.get("prompt_tags")

    @property
    def last_compared_at(self) -> str | None:
        return self._image_data.get("last_compared_at")

    def is_top(self) -> bool:
        return self._chain.is_top(self._node_id)

    def is_bottom(self) -> bool:
        return self._chain.is_bottom(self._node_id)

    def get_links(
        self, better_than: bool = False, worse_than: bool = False
    ) -> list[NodeProxy]:
        if better_than and worse_than:
            raise ValueError(
                "A node cannot be simultaneously better than and worse than the same node. "
                "Set only one of better_than/worse_than, or neither for all links."
            )
        results: list[str]
        if not better_than and not worse_than:
            results = []
            results.extend(self._chain.get_better_than(self._node_id))
            results.extend(self._chain.get_worse_than(self._node_id))
        elif better_than:
            results = list(self._chain.get_better_than(self._node_id))
        else:
            results = list(self._chain.get_worse_than(self._node_id))
        seen: set[str] = set()
        unique: list[NodeProxy] = []
        r: str
        for r in results:
            if r not in seen:
                seen.add(r)
                unique.append(NodeProxy(self._chain, r))
        return unique

    def get_chain(self, only_main: bool = True) -> list[ChainProxy]:
        from .chain_proxy import ChainProxy
        if only_main:
            main: tuple[int, list[str]] | None = self._chain.get_node_main_chain(self._node_id)
            if main is None:
                return []
            return [ChainProxy(self._chain, main[0], main[1])]
        else:
            chains: list[tuple[int, list[str]]] = self._chain.get_node_chains(self._node_id)
            return [ChainProxy(self._chain, i, c) for i, c in chains]

    def get_position_in_chain(self) -> int:
        main: tuple[int, list[str]] | None = self._chain.get_node_main_chain(self._node_id)
        if main is None:
            raise ValueError(f"Node {self._node_id} is not in any chain")
        _: int
        chain: list[str]
        _, chain = main
        try:
            return chain.index(self._node_id)
        except ValueError:
            raise ValueError(f"Node {self._node_id} not found in its own chain")

    def get_component(self) -> ComponentProxy | None:
        from .component_proxy import ComponentProxy
        comp_id: int | None = self._chain.get_component_id(self._node_id)
        if comp_id is None:
            return None
        return ComponentProxy(self._chain, comp_id)

    def __repr__(self) -> str:
        return f"NodeProxy({self._node_id})"
