from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any
import time
from tqdm import tqdm

from ..logger import get_logger

logger = get_logger(__name__)


class ChainManager:
    """Internal data store for the comparison graph. All attributes are private.
    Access is only through CrystalGraph's public API.
    """

    def __init__(self) -> None:
        # -- Core adjacency --
        self._better_than: defaultdict[str, list[str]] = defaultdict(list)
        self._worse_than: defaultdict[str, list[str]] = defaultdict(list)
        self._adjacency: defaultdict[str, set[str]] = defaultdict(set)

        # -- Topology --
        self._top_nodes: list[str] = []
        self._bottom_nodes: list[str] = []

        # -- Components --
        self._node_component: dict[str, int] = {}
        self._component_members: dict[int, list[str]] = {}

        # -- Tracking --
        self._all_filenames: set[str] = set()
        self._built_at: datetime | None = None
        self._db_comparison_count: int = 0

        # -- Chains --
        self._chains: dict[int, list[str]] = {}
        self._node_to_chains: dict[str, list[tuple[int, list[str]]]] = {}
        self._node_main_chain: dict[str, tuple[int, list[str]]] = {}

        self._common_chains: dict[int, tuple[list[str], bool]] = {}
        self._node_chains: dict[str, dict[int, bool]] = {}

    # ==================================================================
    # Public accessors
    # ==================================================================

    def get_all_filenames(self) -> set[str]:
        return self._all_filenames

    def get_top_nodes(self) -> list[str]:
        return self._top_nodes

    def get_bottom_nodes(self) -> list[str]:
        return self._bottom_nodes

    def get_better_than(self, node_id: str) -> list[str]:
        return self._better_than.get(node_id, [])

    def get_worse_than(self, node_id: str) -> list[str]:
        return self._worse_than.get(node_id, [])

    def is_top(self, node_id: str) -> bool:
        return node_id in self._top_nodes

    def is_bottom(self, node_id: str) -> bool:
        return node_id in self._bottom_nodes

    def get_component_id(self, node_id: str) -> int | None:
        return self._node_component.get(node_id)

    def get_component_members(self, comp_id: int) -> list[str]:
        return self._component_members.get(comp_id, [])

    def get_component_count(self) -> int:
        return len(self._component_members)

    def get_built_at(self) -> datetime | None:
        return self._built_at

    def set_built_at(self, dt: datetime | None) -> None:
        self._built_at = dt

    def get_db_comparison_count(self) -> int:
        return self._db_comparison_count

    def set_db_comparison_count(self, count: int) -> None:
        self._db_comparison_count = count

    # ==================================================================
    # Build (full rebuild from comparison list)
    # ==================================================================

    def build(
        self,
        comparisons: list[dict[str, Any]],
        all_filenames: set[str] | None = None,
    ) -> None:
        self._better_than.clear()
        self._worse_than.clear()
        self._adjacency.clear()

        graph_filenames: set[str] = set()
        # with tqdm(comparisons, desc="Processing comparisons", unit="comp") as pbar:
        for comp in comparisons:
            a: str = comp["filename_a"]
            b: str = comp["filename_b"]
            winner: str = comp["winner"]
            loser: str

            if winner == a:
                loser = b
            else:
                loser = a

            if winner not in self._better_than[loser]:
                self._better_than[loser].append(winner)
            if loser not in self._worse_than[winner]:
                self._worse_than[winner].append(loser)

            self._adjacency[a].add(b)
            self._adjacency[b].add(a)
            graph_filenames.add(a)
            graph_filenames.add(b)

        self._all_filenames = (
            all_filenames if all_filenames is not None else graph_filenames
        )
        self._identify_top_bottom()
        self._build_components()
        self._build_chains()
        # self._compute_chains_naive()

    # ==================================================================
    # Incremental update (single comparison, no DB)
    # ==================================================================

    def apply_comparison(self, winner: str, loser: str) -> None:
        _t: float = time.time()
        self._all_filenames.update([winner, loser])

        # -- 1. Add edge --
        if loser not in self._worse_than[winner]:
            self._worse_than[winner].append(loser)
        if winner not in self._better_than[loser]:
            self._better_than[loser].append(winner)
        self._adjacency[winner].add(loser)
        self._adjacency[loser].add(winner)

        # -- 2. Update top_nodes / bottom_nodes --
        if winner in self._bottom_nodes and self._worse_than[winner]:
            self._bottom_nodes.remove(winner)
        if loser in self._top_nodes and self._better_than[loser]:
            self._top_nodes.remove(loser)
        if not self._worse_than[loser] and loser not in self._bottom_nodes:
            self._bottom_nodes.append(loser)
        if not self._better_than[winner] and winner not in self._top_nodes:
            self._top_nodes.append(winner)

        # -- 3. Merge components if different --
        cw: int | None = self._node_component.get(winner)
        cl: int | None = self._node_component.get(loser)
        if cw is not None and cl is not None and cw != cl:
            self._merge_components(cw, cl)
        elif cw is None and cl is None:
            new_id: int = max(self._component_members.keys(), default=-1) + 1
            self._component_members[new_id] = [winner, loser]
            self._node_component[winner] = new_id
            self._node_component[loser] = new_id
        elif cw is None and cl is not None:
            self._node_component[winner] = cl
            self._component_members[cl].append(winner)
        elif cl is None and cw is not None:
            self._node_component[loser] = cw
            self._component_members[cw].append(loser)

        # -- 4. Recompute chains from scratch --
        # self._compute_chains_naive()
        # self._build_chains()

        # -- 5. Bookkeeping --
        self._db_comparison_count += 1
        self._built_at = datetime.now(timezone.utc)

    def _merge_components(self, keep_id: int, remove_id: int) -> None:
        """Merge two components by absorbing the smaller into the larger."""
        if len(self._component_members.get(keep_id, [])) < len(
            self._component_members.get(remove_id, [])
        ):
            keep_id, remove_id = remove_id, keep_id
        with tqdm(
            self._component_members.get(remove_id, []),
            desc="Merging components",
            unit="node",
        ) as pbar:
            for node in pbar:
                self._node_component[node] = keep_id
        self._component_members[keep_id].extend(
            self._component_members.pop(remove_id, [])
        )

    # ==================================================================
    # Internal build helpers
    # ==================================================================

    def _identify_top_bottom(self) -> None:
        # with tqdm(
        #     self._all_filenames, desc="Identifying top nodes", unit="node"
        # ) as pbar:
        self._top_nodes = [n for n in self._all_filenames if not self._better_than[n]]
        # with tqdm(
        #     self._all_filenames, desc="Identifying bottom nodes", unit="node"
        # ) as pbar:
        self._bottom_nodes = [n for n in self._all_filenames if not self._worse_than[n]]

    def _build_components(self) -> None:
        visited: set[str] = set()
        self._node_component = {}
        self._component_members = {}
        comp_id: int = 0
        # with tqdm(self._all_filenames, desc="Building components", unit="node") as pbar:
        for filename in self._all_filenames:
            if filename in visited:
                continue
            queue: deque[str] = deque([filename])
            visited.add(filename)
            members: list[str] = []
            while queue:
                current: str = queue.popleft()
                members.append(current)
                neighbor: str
                for neighbor in self._adjacency.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            self._component_members[comp_id] = members
            member: str
            for member in members:
                self._node_component[member] = comp_id
            comp_id += 1

    def _build_chains(self) -> None:
        self._node_main_chain = {}
        self._chains = {}
        self._build_all_chains()

        assigned_chains: list[int] = []

        # define main chains

        with tqdm(
            total=len(self._node_chains.items()),
            desc="Setting main chains",
            unit="nodes",
        ):
            for node, node_chains in self._node_chains.items():
                longest_chain_length = -1
                longest_chain_id = -1
                for chain_id in node_chains.keys():
                    length: int = len(self._common_chains[chain_id][0])
                    if length > longest_chain_length:
                        longest_chain_length: int = length
                        longest_chain_id: int = chain_id

                if longest_chain_id == -1:
                    # orphan/new images
                    continue
                self._node_chains[node][longest_chain_id] = True

                chain_list: list[str] = self._common_chains[longest_chain_id][0]
                self._node_main_chain[node] = (longest_chain_id, chain_list)

                if longest_chain_id in assigned_chains:
                    continue

                self._common_chains[longest_chain_id] = (
                    chain_list,
                    True,
                )

                self._chains[longest_chain_id] = chain_list
                assigned_chains.append(longest_chain_id)

    def _build_all_chains(self) -> None:
        # chain id,[filename list,is main chain]
        self._common_chains: dict[int, tuple[list[str], bool]] = {}
        # filename,chain list [id chain, is main for this node]
        self._node_chains: dict[str, dict[int, bool]] = {}
        current_node: str = self._top_nodes[0]
        remaining_nodes: list[str] = list(
            set(self._top_nodes).union(self._all_filenames)
        )

        last_chain_created = 0

        with tqdm(
            total=len(remaining_nodes),
            desc="Building all chains from nodes",
            unit="Node",
        ) as pbar:
            while remaining_nodes:
                current_node = remaining_nodes.pop()
                if not current_node in self._node_chains:
                    self._node_chains[current_node] = {}

                # append current node to existing better chains
                if current_node in self._better_than:
                    better: list[str] = self._better_than[current_node]
                    for b in better:
                        if b not in self._node_chains:
                            self._node_chains[b] = {}
                        last_chain_created += 1
                        self._common_chains[last_chain_created] = ([b], False)
                        self._node_chains[b][last_chain_created] = False

                        chain_list: dict[int, bool] = self._node_chains[b]
                        for chain_id in chain_list.keys():
                            self._node_chains[current_node][chain_id] = False
                            self._common_chains[chain_id][0].append(current_node)

                # append existing worse nodes to current node's chains
                if current_node in self._worse_than:
                    worse: list[str] = self._worse_than[current_node]
                    for w in worse:
                        if w not in self._node_chains:
                            self._node_chains[w] = {}
                        last_chain_created += 1
                        self._common_chains[last_chain_created] = ([w], False)
                        self._node_chains[w][last_chain_created] = False

                        chain_list: dict[int, bool] = self._node_chains[w]
                        for chain_id in chain_list.keys():
                            self._node_chains[current_node][chain_id] = False
                            self._common_chains[chain_id][0].append(current_node)

                pbar.update(1)
                pbar.set_description(f"chains created: {last_chain_created}")

    # def _compute_chains_naive(self) -> None:
    #     """Naive chain computation: iterate from top nodes, build longest paths."""
    #     self._chains = []
    #     self._node_to_chains = {}
    #     self._node_main_chain = {}

    #     if not self._all_filenames:
    #         return

    #     covered: set[str] = set()
    #     all_sorted: list[str] = sorted(self._all_filenames)

    #     def _find_best_start() -> str | None:
    #         # Prefer top nodes first
    #         node: str
    #         for node in self._top_nodes:
    #             if node not in covered and self._worse_than.get(node):
    #                 return node
    #         # Then any uncovered node with outgoing edges
    #         for node in all_sorted:
    #             if node not in covered and self._worse_than.get(node):
    #                 return node
    #         # Finally any uncovered node
    #         for node in all_sorted:
    #             if node not in covered:
    #                 return node
    #         return None

    #     def _build_chain(start: str) -> list[str]:
    #         chain: list[str] = []
    #         in_chain: set[str] = set()
    #         cur: str = start

    #         # First, go backwards to find the true start (topmost ancestor)
    #         stack: list[str] = [cur]
    #         while stack:
    #             node: str = stack[-1]
    #             preds: list[str] = [
    #                 p for p in self._better_than.get(node, []) if p not in in_chain
    #             ]
    #             if not preds:
    #                 break
    #             # Pick the predecessor that's furthest from any bottom (heuristic)
    #             best_pred: str = max(
    #                 preds, key=lambda p: len(self._worse_than.get(p, []))
    #             )
    #             if best_pred in covered:
    #                 break
    #             stack.append(best_pred)

    #         # Pop the stack to build chain from top
    #         while stack:
    #             node = stack.pop()
    #             if node not in in_chain and node not in covered:
    #                 chain.append(node)
    #                 in_chain.add(node)
    #                 covered.add(node)

    #         # Now go forward from the last node
    #         cur = chain[-1] if chain else start
    #         while True:
    #             successors: list[str] = [
    #                 s
    #                 for s in self._worse_than.get(cur, [])
    #                 if s not in in_chain and s not in covered
    #             ]
    #             if not successors:
    #                 break
    #             # Pick successor with most uncovered downstream nodes
    #             best: str = max(
    #                 successors, key=lambda s: len(self._worse_than.get(s, []))
    #             )
    #             chain.append(best)
    #             in_chain.add(best)
    #             covered.add(best)
    #             cur = best

    #         return chain

    #     total: int = len(self._all_filenames)
    #     with tqdm(total=total, desc="Building chains", unit="node") as pbar:
    #         while len(covered) < total:
    #             start: str | None = _find_best_start()
    #             if start is None:
    #                 break
    #             prev_covered: int = len(covered)
    #             chain: list[str] = _build_chain(start)
    #             if chain:
    #                 self._chains.append(chain)
    #                 pbar.update(len(covered) - prev_covered)

    #     # Build node -> chains index
    #     i: int
    #     chain: list[str]
    #     with tqdm(
    #         enumerate(self._chains),
    #         desc="Indexing chains",
    #         total=len(self._chains),
    #         unit="chain",
    #     ) as pbar:
    #         for i, chain in pbar:
    #             node: str
    #             for node in chain:
    #                 if node not in self._node_to_chains:
    #                     self._node_to_chains[node] = []
    #                 self._node_to_chains[node].append((i, chain))

    #     # Assign main chain = longest chain containing each node
    #     with tqdm(
    #         self._all_filenames, desc="Assigning main chains", unit="node"
    #     ) as pbar:
    #         for node in pbar:
    #             chains: list[tuple[int, list[str]]] = self._node_to_chains.get(node, [])
    #             if chains:
    #                 self._node_main_chain[node] = max(chains, key=lambda x: len(x[1]))

    def get_chains(self) -> dict[int, list[str]]:
        return self._chains

    def get_node_chains(self, node_id: str) -> list[tuple[int, list[str]]]:
        return self._node_to_chains.get(node_id, [])

    def get_node_main_chain(self, node_id: str) -> tuple[int, list[str]] | None:
        return self._node_main_chain.get(node_id)

    def get_min_chain_count(self) -> int:
        return len(self._chains.items())

    def _can_reach(
        self,
        start: str,
        end: str,
        skip_edges: set[tuple[str, str]] | None = None,
        max_depth: int = 10,
    ) -> bool:
        _start = time.perf_counter()
        if start not in self._all_filenames or end not in self._all_filenames:
            return False
        if start == end:
            return True

        # Quick component-based rejection
        c_start = self._node_component.get(start)
        c_end = self._node_component.get(end)
        if c_start is not None and c_end is not None and c_start != c_end:
            return False

        # Quick rejection: bottom nodes can't reach anyone
        if not self._worse_than.get(start):
            return False

        # Quick rejection: no one can reach a top node
        if not self._better_than.get(end):
            return False

        # Fast same-chain check (only safe without edge constraints)
        if not skip_edges:
            same_chain, start_before_end = self._check_same_chain(start, end)
            if same_chain:
                return True
            # if same_chain:
            #     return start_before_end

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        depth = -1
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for w in self._worse_than.get(current, []):
                if skip_edges and (current, w) in skip_edges:
                    continue
                if w == end:
                    logger.debug(f"can reach at depth {depth}", start_timer=_start)
                    return True
                if w not in visited:
                    visited.add(w)
                    queue.append((w, depth + 1))
        logger.debug(f"{start[:5]} cannot reach {end[:5]} up to depth {depth}", start_timer=_start)
        return False

    def _check_same_chain(self, u: str, v: str) -> tuple[bool, bool]:
        """Check if two nodes share a chain in the minimum chain cover.

        Returns (same_chain, u_before_v). Only valid when chains are cached.
        """
        chains_u = self._node_chains.get(u)
        chains_v = self._node_chains.get(v)
        if not chains_u or not chains_v:
            return (False, False)
        chain_list_u = list(chains_u.keys())
        chain_list_v = list(chains_v.keys())
        common_chain = []
        for item in chain_list_u:
            if item in chain_list_v:
                common_chain = self._common_chains[item][0]
                break

        if len(common_chain) == 0:
            return (False, False)

        u_position = common_chain.index(u)
        v_position = common_chain.index(v)
        return (True, u_position < v_position)
