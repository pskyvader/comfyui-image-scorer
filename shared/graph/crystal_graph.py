from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from tqdm import tqdm
import time


from ...external_modules.database_structure.images_table import (
    get_all_images,
)
from ...external_modules.database_structure.comparisons_table import (
    get_all_comparisons,
    get_total_comparisons,
)

from ..logger import get_logger, ModuleLogger
from .chain_manager import ChainManager
from .node_proxy import NodeProxy
from .chain_proxy import ChainProxy
from .component_proxy import ComponentProxy

logger: ModuleLogger = get_logger(__name__)

# Type aliases for chain mapping structures
NodeTuple = tuple[NodeProxy, bool]
ChainTuple = tuple[ChainProxy, list[NodeTuple]]
ChainDict = dict[int, ChainTuple]


class CrystalGraph:
    """Main graph API. All access through get_* methods returning proxy objects."""

    def __init__(self) -> None:
        self._chain: ChainManager = ChainManager()
        self._images: dict[str, dict[str, Any]] = {}
        self._comparisons: list[dict[str, Any]] = []
        self._chain_map: dict[int, ChainDict] | None = None
        self._rebuilding: bool = False

    # -- Lifecycle ------------------------------------------------------
    def get_node_chain_length(self, filename: str) -> int:
        main: tuple[int, list[str]] | None = self._chain.get_node_main_chain(filename)
        if main is None:
            return 0
        return len(main[1])

    def rebuild_from_database(
        self,
        images: list[dict[str, Any]] | None = None,
        comparisons: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._rebuilding:
            logger.warning("Already rebuilding, skipping nested call")
            return
        self._rebuilding = True
        logger.info("Rebuilding chain from database...")

        if images is None:
            images = get_all_images()
        if comparisons is None:
            comparisons = get_all_comparisons()

        self._images = {img["filename"]: img for img in images}
        self._comparisons = comparisons
        self._chain.set_db_comparison_count(len(comparisons))
        self._chain.set_built_at(datetime.now(timezone.utc))

        all_filenames: set[str] = set(self._images.keys())
        comp: dict[str, Any]
        # with tqdm(comparisons, desc="Collecting filenames", unit="comp") as pbar:
        for comp in comparisons:
            all_filenames.add(comp["filename_a"])
            all_filenames.add(comp["filename_b"])

        self._chain.build(comparisons, all_filenames=all_filenames)
        self._chain_map = None
        self.get_chains_map()
        self._rebuilding = False

    def apply_comparison(self, winner: str, loser: str) -> None:
        self._chain.apply_comparison(winner, loser)

    # -- Cache ----------------------------------------------------------

    def is_cache_stale(self) -> bool:
        if self._chain.get_built_at() is None:
            return True

        return get_total_comparisons() != self._chain.get_db_comparison_count()

    # -- Node lookups ---------------------------------------------------

    def get_node(self, node_id: str | None = None) -> NodeProxy | None:
        if node_id is None or node_id not in self._chain.get_all_filenames():
            return None
        image_data: dict[str, Any] | None = self._images.get(node_id)
        return NodeProxy(self._chain, node_id, image_data)

    def get_all_nodes(
        self, only_top: bool = False, only_bottom: bool = False
    ) -> list[NodeProxy]:
        if only_top and only_bottom:
            raise ValueError("only_top and only_bottom cannot both be True")
        if only_top:
            return [
                NodeProxy(self._chain, n, self._images.get(n))
                for n in self._chain.get_top_nodes()
            ]
        if only_bottom:
            return [
                NodeProxy(self._chain, n, self._images.get(n))
                for n in self._chain.get_bottom_nodes()
            ]
        return [
            NodeProxy(self._chain, n, self._images.get(n))
            for n in self._chain.get_all_filenames()
        ]

    # -- Chain lookups --------------------------------------------------

    def get_chain(
        self, node_id: str | None = None, chain_id: int | None = None
    ) -> ChainProxy | None:
        if (node_id is None) == (chain_id is None):
            raise ValueError("Exactly one of node_id or chain_id is required")
        if node_id is not None:
            if node_id not in self._chain.get_all_filenames():
                return None
            main: tuple[int, list[str]] | None = self._chain.get_node_main_chain(
                node_id
            )
            if main is None:
                return None
            return ChainProxy(self._chain, main[0], main[1])
        if chain_id is not None:
            chains: dict[int, list[str]] = self._chain.get_chains()
            if chain_id < 0 or chain_id not in chains:
                return None
            return ChainProxy(self._chain, chain_id, chains[chain_id])
        return None

    def get_all_chains(
        self,
        min_length: int = 0,
        sort_order: str = "desc",
    ) -> list[tuple[ChainProxy, list[NodeTuple]]]:
        _start = time.perf_counter()
        result: list[tuple[ChainProxy, list[NodeTuple]]] = []
        length_data: ChainDict
        chain: ChainProxy
        node_list: list[NodeTuple]
        for length_data in self.get_chains_map().values():
            for chain, node_list in length_data.values():
                result.append((chain, node_list))
        if min_length > 0:
            result = [c for c in result if min_length <= c[0].length]
        result.sort(key=lambda c: c[0].length, reverse=(sort_order != "asc"))
        logger.debug(f"all chains: {len(result)}", start_timer=_start)
        return result

    def get_component(
        self,
        node_id: str | None = None,
        component_id: int | None = None,
        chain_id: int | None = None,
    ) -> ComponentProxy | None:
        n_specified: int = sum(
            1 for x in (node_id, component_id, chain_id) if x is not None
        )
        if n_specified != 1:
            raise ValueError(
                "Exactly one of node_id, component_id, or chain_id is required"
            )
        if node_id is not None:
            cid: int | None = self._chain.get_component_id(node_id)
            if cid is None:
                return None
            return ComponentProxy(self._chain, cid)
        if component_id is not None:
            if component_id not in self._chain._component_members:
                return None
            return ComponentProxy(self._chain, component_id)
        if chain_id is not None:
            chain: ChainProxy | None = self.get_chain(chain_id=chain_id)
            if chain is None or not chain._nodes:
                return None
            cid = self._chain.get_component_id(chain._nodes[0])
            if cid is None:
                return None
            return ComponentProxy(self._chain, cid)
        return None

    def get_all_components(self) -> list[ComponentProxy]:
        return [
            ComponentProxy(self._chain, cid) for cid in self._chain._component_members
        ]

    # -- Links ----------------------------------------------------------

    def get_all_links(self) -> list[tuple[NodeProxy, NodeProxy]]:
        result: list[tuple[NodeProxy, NodeProxy]] = []
        seen: set[tuple[str, str]] = set()
        node_id: str
        loser: str
        key: tuple[str, str]
        with tqdm(
            self._chain.get_all_filenames(), desc="Collecting links", unit="node"
        ) as pbar:
            for node_id in pbar:
                for loser in self._chain.get_worse_than(node_id):
                    key = (node_id, loser)
                    if key not in seen:
                        seen.add(key)
                        result.append(
                            (
                                NodeProxy(
                                    self._chain, node_id, self._images.get(node_id)
                                ),
                                NodeProxy(self._chain, loser, self._images.get(loser)),
                            )
                        )
        return result

    # -- Stats ----------------------------------------------------------

    def get_graph_stats(self) -> dict[str, Any]:
        chains: list[list[str]] = list((self._chain.get_chains()).values())
        built_at: datetime | None = self._chain.get_built_at()
        return {
            "total_images": len(self._images) or len(self._chain.get_all_filenames()),
            "total_comparisons": self._chain.get_db_comparison_count(),
            "total_components": self._chain.get_component_count(),
            "total_chains": len(chains),
            "longest_chain_depth": max((len(c) for c in chains), default=0),
            "top_nodes_count": len(self._chain.get_top_nodes()),
            "bottom_nodes_count": len(self._chain.get_bottom_nodes()),
            "built_at": built_at.isoformat() if built_at is not None else None,
        }

    def are_in_same_path(self, img1: str, img2: str) -> bool:
        if (
            img1 not in self._chain.get_all_filenames()
            or img2 not in self._chain.get_all_filenames()
        ):
            return False
        if self._chain._can_reach(img1, img2):
            return True
        if self._chain._can_reach(img2, img1):
            return True
        return False

    def get_chains_map(self) -> dict[int, ChainDict]:
        if self._chain_map is not None:
            return self._chain_map

        min_chains: dict[int, list[str]] = self._chain.get_chains()

        # Group chains by length
        chain_map: dict[int, list[tuple[int, list[str]]]] = {}
        i: int
        chain: list[str]

        with tqdm(
            desc="Grouping chains by length",
            total=len(min_chains),
            unit="chain",
        ) as pbar:
            for i, chain in min_chains.items():
                length: int = len(chain)
                if length not in chain_map:
                    chain_map[length] = []
                chain_map[length].append((i, chain))

        # Build main chains mapping
        all_nodes: list[NodeProxy] = self.get_all_nodes()
        main_chains: dict[int, list[str]] = {}
        node: NodeProxy
        main: tuple[int, list[str]] | None
        chain_id: int
        with tqdm(all_nodes, desc="Mapping nodes to chains", unit="node") as pbar:
            for node in pbar:
                main = self._chain.get_node_main_chain(node.filename)
                if main is not None:
                    chain_id = main[0]
                    if chain_id not in main_chains:
                        main_chains[chain_id] = []
                    main_chains[chain_id].append(node.filename)

        # Build final map
        final_map: dict[int, ChainDict] = {}
        errors: list[int] = []
        chain_list: list[tuple[int, list[str]]]
        local_main_chains: list[str]
        chain_proxy: ChainProxy
        final_chain: list[NodeTuple]
        node_name: str
        is_main_node: bool
        node_proxy: NodeProxy
        with tqdm(
            chain_map.items(),
            desc="Building final chain map",
            total=len(chain_map),
            unit="chain lengths",
        ) as pbar:
            for length, chain_list in pbar:
                final_map[length] = {}
                for chain_id, chain_nodes in chain_list:
                    if chain_id not in main_chains:
                        errors.append(chain_id)
                        continue

                    local_main_chains = main_chains[chain_id]
                    chain_proxy = ChainProxy(self._chain, chain_id, chain_nodes)

                    final_chain = []
                    for node_name in chain_nodes:
                        is_main_node = node_name in local_main_chains
                        node_proxy = NodeProxy(
                            self._chain, node_name, self._images.get(node_name)
                        )
                        final_chain.append((node_proxy, is_main_node))

                    final_map[length][chain_id] = (chain_proxy, final_chain)

        self._chain_map = final_map
        if len(errors) > 0:
            logger.warning(
                f"Chain mapping completed with {len(errors)} non real main chains"
                # f": {errors}"
            )
        return self._chain_map


crystal_graph: CrystalGraph = CrystalGraph()
crystal_graph.rebuild_from_database()
