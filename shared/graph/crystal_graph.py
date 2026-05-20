from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any
import time
import logging
from tqdm import tqdm

logger: logging.Logger = logging.getLogger(__name__)


class ChainManager:
    """Internal data store for the comparison graph. All attributes are private.
    Access is only through CrystalGraph's public API.
    """

    def __init__(self) -> None:
        # -- Core adjacency --
        self._better_than: defaultdict[str, list[str]] = defaultdict(list)
        self._worse_than: defaultdict[str, list[str]] = defaultdict(list)
        self._adjacency: defaultdict[str, set[str]] = defaultdict(set)

        # -- Chain length: separate ending/starting for incremental updates --
        self._chain_length_ending: dict[str, int] = {}
        self._chain_length_starting: dict[str, int] = {}
        self._chain_length: dict[str, int] = {}
        self._nodes_by_length: dict[int, list[str]] = defaultdict(list)

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

        # -- Caches --
        self._min_chain_cache: list[list[str]] | None = None
        self._node_to_chains_cache: dict[str, list[tuple[int, list[str]]]] | None = None

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
        for comp in comparisons:
            a = comp["filename_a"]
            b = comp["filename_b"]
            winner = comp["winner"]

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
        self._calculate_chain_lengths()
        self._build_components()
        self._min_chain_cache = None
        self._prewarm()

    # ==================================================================
    # Incremental update (single comparison, no DB)
    # ==================================================================

    def apply_comparison(self, winner: str, loser: str) -> None:
        _t = time.time()
        _s = _t
        self._all_filenames.update([winner, loser])

        # -- 1. Add edge --
        if loser not in self._worse_than[winner]:
            self._worse_than[winner].append(loser)
        if winner not in self._better_than[loser]:
            self._better_than[loser].append(winner)
        self._adjacency[winner].add(loser)
        self._adjacency[loser].add(winner)
        # logger.info(f"  [APPLY] add_edge: {time.time()-_s:.4f}s")
        _s = time.time()

        # Initialize ending/starting for new nodes
        if winner not in self._chain_length_ending:
            self._chain_length_ending[winner] = 0
            self._chain_length_starting[winner] = 0
        if loser not in self._chain_length_ending:
            self._chain_length_ending[loser] = 0
            self._chain_length_starting[loser] = 0

        # -- 2. Update top_nodes / bottom_nodes --
        if winner in self._bottom_nodes and self._worse_than[winner]:
            self._bottom_nodes.remove(winner)
        if loser in self._top_nodes and self._better_than[loser]:
            self._top_nodes.remove(loser)
        if not self._worse_than[loser] and loser not in self._bottom_nodes:
            self._bottom_nodes.append(loser)
        if not self._better_than[winner] and winner not in self._top_nodes:
            self._top_nodes.append(winner)
        # logger.info(f"  [APPLY] top_bottom: {time.time()-_s:.4f}s")
        _s = time.time()

        # -- 3. Forward propagate longest_starting from winner --
        touched: set[str] = set()
        new_starting = 1 + self._chain_length_starting.get(loser, 0)
        old_starting = self._chain_length_starting.get(winner, 0)
        if new_starting > old_starting:
            self._chain_length_starting[winner] = new_starting
            touched.add(winner)
            queue = deque([winner])
            visited = {winner}
            while queue:
                node = queue.popleft()
                for pred in self._better_than.get(node, []):
                    expected = 1 + self._chain_length_starting[node]
                    if expected > self._chain_length_starting.get(pred, 0):
                        self._chain_length_starting[pred] = expected
                        touched.add(pred)
                        if pred not in visited:
                            visited.add(pred)
                            queue.append(pred)
        # logger.info(
        #     f"  [APPLY] forward_propagate ({len(touched)} nodes): {time.time()-_s:.4f}s"
        # )
        _s = time.time()

        # -- 4. Backward propagate longest_ending from loser --
        new_ending = 1 + self._chain_length_ending.get(winner, 0)
        old_ending = self._chain_length_ending.get(loser, 0)
        if new_ending > old_ending:
            self._chain_length_ending[loser] = new_ending
            touched.add(loser)
            queue = deque([loser])
            visited = {loser}
            while queue:
                node = queue.popleft()
                for succ in self._worse_than.get(node, []):
                    expected = 1 + self._chain_length_ending[node]
                    if expected > self._chain_length_ending.get(succ, 0):
                        self._chain_length_ending[succ] = expected
                        touched.add(succ)
                        if succ not in visited:
                            visited.add(succ)
                            queue.append(succ)
        # logger.info(
        #     f"  [APPLY] backward_propagate ({len(touched)} nodes): {time.time()-_s:.4f}s"
        # )
        _s = time.time()

        # -- 5. Recompute chain_length for touched nodes --
        for node in touched:
            ending = self._chain_length_ending.get(node, 0)
            starting = self._chain_length_starting.get(node, 0)
            old_val = self._chain_length.get(node, 0)
            new_val = ending + starting
            if new_val != old_val:
                if old_val > 0 and node in self._nodes_by_length.get(old_val, []):
                    self._nodes_by_length[old_val].remove(node)
                self._chain_length[node] = new_val
                self._nodes_by_length[new_val].append(node)
        for node in (winner, loser):
            if node not in self._chain_length:
                ending = self._chain_length_ending.get(node, 0)
                starting = self._chain_length_starting.get(node, 0)
                self._chain_length[node] = ending + starting
                self._nodes_by_length[self._chain_length[node]].append(node)
        # logger.info(
        #     f"  [APPLY] recompute_chain_length ({len(touched)} nodes): {time.time()-_s:.4f}s"
        # )
        _s = time.time()

        # -- 6. Merge components if different --
        cw = self._node_component.get(winner)
        cl = self._node_component.get(loser)
        if cw is not None and cl is not None and cw != cl:
            self._merge_components(cw, cl)
        elif cw is None and cl is None:
            new_id = max(self._component_members.keys(), default=-1) + 1
            self._component_members[new_id] = [winner, loser]
            self._node_component[winner] = new_id
            self._node_component[loser] = new_id
        elif cw is None:
            self._node_component[winner] = cl
            self._component_members[cl].append(winner)
        elif cl is None:
            self._node_component[loser] = cw
            self._component_members[cw].append(loser)
        # logger.info(f"  [APPLY] merge_components: {time.time()-_s:.4f}s")
        # _s = time.time()

        # -- 7. Bookkeeping --
        self._db_comparison_count += 1
        self._built_at = datetime.now(timezone.utc)
        logger.info(f"  [APPLY] TOTAL: {time.time()-_t:.4f}s")

    def _merge_components(self, keep_id: int, remove_id: int) -> None:
        """Merge two components by absorbing the smaller into the larger."""
        if len(self._component_members.get(keep_id, [])) < len(
            self._component_members.get(remove_id, [])
        ):
            keep_id, remove_id = remove_id, keep_id
        for node in self._component_members.get(remove_id, []):
            self._node_component[node] = keep_id
        self._component_members[keep_id].extend(
            self._component_members.pop(remove_id, [])
        )

    # ==================================================================
    # Internal build helpers
    # ==================================================================

    def _identify_top_bottom(self) -> None:
        self._top_nodes = [n for n in self._all_filenames if not self._better_than[n]]
        self._bottom_nodes = [n for n in self._all_filenames if not self._worse_than[n]]

    def _calculate_chain_lengths(self) -> None:
        """Chain length = longest_ending_at + longest_starting_from. Iterative DP."""
        # Forward pass (longest path from node to sink)
        self._chain_length_starting = {}
        outdegree = {n: len(self._worse_than.get(n, [])) for n in self._all_filenames}
        queue = deque([n for n in self._all_filenames if outdegree[n] == 0])
        while queue:
            node = queue.popleft()
            best = 0
            for succ in self._worse_than.get(node, []):
                if self._chain_length_starting.get(succ, 0) + 1 > best:
                    best = self._chain_length_starting.get(succ, 0) + 1
            self._chain_length_starting[node] = best
            for pred in self._better_than.get(node, []):
                outdegree[pred] -= 1
                if outdegree[pred] == 0:
                    queue.append(pred)

        # Backward pass (longest path from source to node)
        self._chain_length_ending = {}
        indegree = {n: len(self._better_than.get(n, [])) for n in self._all_filenames}
        queue = deque([n for n in self._all_filenames if indegree[n] == 0])
        while queue:
            node = queue.popleft()
            best = 0
            for pred in self._better_than.get(node, []):
                if self._chain_length_ending.get(pred, 0) + 1 > best:
                    best = self._chain_length_ending.get(pred, 0) + 1
            self._chain_length_ending[node] = best
            for succ in self._worse_than.get(node, []):
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    queue.append(succ)

        self._chain_length = {}
        self._nodes_by_length.clear()
        for node in self._all_filenames:
            ending = self._chain_length_ending.get(node, 0)
            starting = self._chain_length_starting.get(node, 0)
            has_comparisons = node in self._worse_than or node in self._better_than
            if not has_comparisons:
                self._chain_length[node] = 0
            else:
                self._chain_length[node] = ending + starting
            self._nodes_by_length[self._chain_length[node]].append(node)
        # Invalidate node-to-chains cache since chains changed
        self._node_to_chains_cache = None

    def _build_components(self) -> None:
        visited: set[str] = set()
        self._node_component = {}
        self._component_members = {}
        comp_id = 0
        for filename in self._all_filenames:
            if filename in visited:
                continue
            queue = deque([filename])
            visited.add(filename)
            members: list[str] = []
            while queue:
                current = queue.popleft()
                members.append(current)
                for neighbor in self._adjacency.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            self._component_members[comp_id] = members
            for member in members:
                self._node_component[member] = comp_id
            comp_id += 1

    def _prewarm(self) -> None:
        """Pre-compute the min chain cover after build so it's cached for first request."""
        _t = time.time()
        self.compute_min_chains()
        logger.info(
            f"[CHAIN] min_chain_count prewarmed ({len(self._min_chain_cache)} chains): {time.time()-_t:.3f}s"
        )

    # ==================================================================
    # Minimum chain cover
    # ==================================================================

    def _compute_depth_from_source(self) -> dict[str, int]:
        depth: dict[str, int] = {n: 0 for n in self._all_filenames}
        indegree = {n: len(self._better_than.get(n, [])) for n in self._all_filenames}
        queue = deque([n for n in self._all_filenames if indegree[n] == 0])
        while queue:
            node = queue.popleft()
            for succ in self._worse_than.get(node, []):
                if depth[node] + 1 > depth[succ]:
                    depth[succ] = depth[node] + 1
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    queue.append(succ)
        return depth

    def _compute_depth_to_sink(self) -> dict[str, int]:
        depth: dict[str, int] = {n: 0 for n in self._all_filenames}
        outdegree = {n: len(self._worse_than.get(n, [])) for n in self._all_filenames}
        queue = deque([n for n in self._all_filenames if outdegree[n] == 0])
        while queue:
            node = queue.popleft()
            for pred in self._better_than.get(node, []):
                if depth[node] + 1 > depth[pred]:
                    depth[pred] = depth[node] + 1
                outdegree[pred] -= 1
                if outdegree[pred] == 0:
                    queue.append(pred)
        return depth

    def _do_min_chains(self) -> list[list[str]]:
        if not self._all_filenames:
            return []
        _t = time.time()
        _s = _t
        depth_from_source = self._compute_depth_from_source()
        logger.info(f"    [MINCHAIN] depth_from_source: {time.time()-_s:.4f}s")
        _s = time.time()
        depth_to_sink = self._compute_depth_to_sink()
        logger.info(f"    [MINCHAIN] depth_to_sink: {time.time()-_s:.4f}s")
        _s = time.time()
        by_depth: dict[int, list[str]] = defaultdict(list)
        for node, d in depth_from_source.items():
            by_depth[d].append(node)
        if not by_depth:
            return []
        graph_nodes = set(self._better_than.keys()) | set(self._worse_than.keys())
        if not graph_nodes:
            return [[n] for n in self._all_filenames]

        covered: set[str] = set()
        chains: list[list[str]] = []
        all_sorted = sorted(
            self._all_filenames, key=lambda n: depth_from_source.get(n, 0)
        )
        logger.info(f"    [MINCHAIN] setup: {time.time()-_s:.4f}s")
        _s = time.time()

        def _count_uncov_downstream(node: str, skip: set[str]) -> int:
            visited: set[str] = set()
            stack = [node]
            count = 0
            while stack and count < 50000:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                if n not in covered and n not in skip:
                    count += 1
                for succ in self._worse_than.get(n, []):
                    if succ not in visited:
                        stack.append(succ)
            return count

        def _find_best_start() -> str | None:
            for node in self._top_nodes:
                if node not in covered and _count_uncov_downstream(node, set()) > 0:
                    return node
            for node in all_sorted:
                if (
                    node not in covered
                    and self._worse_than.get(node)
                    and _count_uncov_downstream(node, set()) > 0
                ):
                    return node
            for node in all_sorted:
                if node not in covered:
                    return node
            return None

        def _build_chain(start: str) -> list[str]:
            chain: list[str] = []
            in_chain: set[str] = set()
            cur = start
            while cur is not None and cur not in in_chain:
                chain.append(cur)
                in_chain.add(cur)
                covered.add(cur)
                successors = self._worse_than.get(cur, [])
                candidates = [s for s in successors if s not in in_chain]
                if not candidates:
                    break

                def score(n: str) -> tuple[int, int]:
                    return (1 if n not in covered else 0, depth_to_sink.get(n, 0))

                candidates.sort(key=score, reverse=True)
                if len(candidates) > 1 and score(candidates[0])[0] == 1:
                    best_score = score(candidates[0])
                    tied = [c for c in candidates if score(c) == best_score]
                    best = (
                        max(tied, key=lambda n: _count_uncov_downstream(n, in_chain))
                        if len(tied) > 1
                        else tied[0]
                    )
                else:
                    best = candidates[0]
                cur = best

            if start not in self._top_nodes:
                cur = start
                while True:
                    preds = [
                        p for p in self._better_than.get(cur, []) if p not in in_chain
                    ]
                    if not preds:
                        break
                    best_pred = max(preds, key=lambda p: depth_from_source.get(p, 0))
                    chain.insert(0, best_pred)
                    in_chain.add(best_pred)
                    covered.add(best_pred)
                    cur = best_pred
                    if cur in self._top_nodes:
                        break
            return chain

        while len(covered) < len(self._all_filenames):
            start = _find_best_start()
            if start is None:
                break
            chain = _build_chain(start)
            if not chain:
                for node in self._all_filenames:
                    if node not in covered:
                        covered.add(node)
                        chains.append([node])
                        break
            else:
                chains.append(chain)
        logger.info(
            f"    [MINCHAIN] greedy_build ({len(chains)} chains): {time.time()-_s:.4f}s"
        )
        _s = time.time()
        logger.info(f"    [MINCHAIN] TOTAL: {time.time()-_t:.4f}s")
        return chains

    def compute_min_chains(self) -> list[list[str]]:
        if self._min_chain_cache is None:
            self._min_chain_cache = self._do_min_chains()
            # Pre-compute node-to-chains index when chains are computed
            self._build_node_to_chains_index()
        return self._min_chain_cache

    def _build_node_to_chains_index(self) -> None:
        """Build reverse index: node_id -> list of (chain_id, chain) tuples."""
        if self._min_chain_cache is None:
            self._node_to_chains_cache = {}
            return
        index: dict[str, list[tuple[int, list[str]]]] = {}
        for i, chain in enumerate(self._min_chain_cache):
            for node in chain:
                if node not in index:
                    index[node] = []
                index[node].append((i, chain))
        self._node_to_chains_cache = index

    def _get_node_chains(self, node_id: str) -> list[tuple[int, list[str]]]:
        """Fast lookup: get all chains containing a node. O(1) after index is built."""
        # Ensure chains and index are computed
        if self._node_to_chains_cache is None:
            self.compute_min_chains()
        return self._node_to_chains_cache.get(node_id, [])

    def _invalidate_min_chain_cache(self) -> None:
        self._min_chain_cache = None
        self._node_to_chains_cache = None

    def get_min_chain_count(self) -> int:
        return len(self.compute_min_chains())

    # ==================================================================
    # Chain enumeration
    # ==================================================================

    def enumerate_all_chains(
        self, max_chains: int = 20000000, max_depth: int = 50000
    ) -> list[list[str]]:
        chains: list[list[str]] = []
        stack: list[tuple[str, list[str]]] = []
        for start in list(self._all_filenames):
            stack.append((start, [start]))
            while stack and len(chains) < max_chains:
                node, path = stack.pop()
                for nb in self._worse_than.get(node, []):
                    if nb in path:
                        continue
                    newpath = path + [nb]
                    if len(newpath) >= 2:
                        chains.append(newpath)
                    if len(newpath) < max_depth:
                        stack.append((nb, newpath))
                if len(chains) >= max_chains:
                    break
        return chains

    def enumerate_top_chains(self) -> list[list[str]]:
        chains: list[list[str]] = []
        for top in self._top_nodes:
            stack = [(top, [top])]
            while stack:
                node, path = stack.pop()
                worse_nodes = self._worse_than.get(node, [])
                if not worse_nodes:
                    chains.append(path)
                else:
                    for w in worse_nodes:
                        if w not in path:
                            stack.append((w, path + [w]))
        return chains

    # ==================================================================
    # Query / traversal
    # ==================================================================

    def _can_reach(
        self,
        start: str,
        end: str,
        skip_edges: set[tuple[str, str]] | None = None,
    ) -> bool:
        if start not in self._all_filenames or end not in self._all_filenames:
            return False
        visited = set()
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current == end:
                return True
            visited.add(current)
            for w in self._worse_than.get(current, []):
                if w not in visited:
                    if skip_edges and (current, w) in skip_edges:
                        continue
                    queue.append(w)
        return False

    def is_redundant(
        self,
        u: str,
        v: str,
        skip_edges: set[tuple[str, str]] | None = None,
    ) -> bool:
        if u not in self._all_filenames or v not in self._all_filenames:
            return False
        all_skip = set()
        if skip_edges:
            all_skip.update(skip_edges)
        all_skip.add((u, v))
        for w in self._worse_than.get(u, []):
            if w == v:
                continue
            if self._can_reach(w, v, all_skip):
                return True
        return False

    def are_in_same_path(self, img1: str, img2: str) -> bool:
        if img1 not in self._all_filenames or img2 not in self._all_filenames:
            return False
        if img1 == img2:
            return True
        if self._can_reach(img1, img2):
            return True
        if self._can_reach(img2, img1):
            return True
        return False

    def get_collapsable_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for length, nodes in self._nodes_by_length.items():
            if len(nodes) > 1:
                top_at_len = [n for n in nodes if not self._better_than[n]]
                bottom_at_len = [n for n in nodes if not self._worse_than[n]]
                if len(top_at_len) > 1:
                    for i in range(0, len(top_at_len) - 1, 2):
                        if i + 1 < len(top_at_len):
                            pairs.append((top_at_len[i], top_at_len[i + 1]))
                if len(bottom_at_len) > 1:
                    for i in range(0, len(bottom_at_len) - 1, 2):
                        if i + 1 < len(bottom_at_len):
                            pairs.append((bottom_at_len[i], bottom_at_len[i + 1]))
        return pairs


# ======================================================================
# Proxy classes — lightweight, ephemeral, hold no mutable state
# ======================================================================


class NodeProxy:
    """Represents one image/node in the graph. Created on demand, zero overhead."""

    def __init__(self, chain: ChainManager, node_id: str) -> None:
        self._chain = chain
        self._node_id = node_id

    @property
    def filename(self) -> str:
        return self._node_id

    def is_top(self) -> bool:
        return self._node_id in self._chain._top_nodes

    def is_bottom(self) -> bool:
        return self._node_id in self._chain._bottom_nodes

    def get_links(
        self, better_than: bool = False, worse_than: bool = False
    ) -> list[NodeProxy]:
        if better_than and worse_than:
            raise ValueError(
                "A node cannot be simultaneously better than and worse than the same node. "
                "Set only one of better_than/worse_than, or neither for all links."
            )
        if not better_than and not worse_than:
            results: list[str] = []
            results.extend(self._chain._better_than.get(self._node_id, []))
            results.extend(self._chain._worse_than.get(self._node_id, []))
        elif better_than:
            results = list(self._chain._better_than.get(self._node_id, []))
        else:
            results = list(self._chain._worse_than.get(self._node_id, []))
        seen: set[str] = set()
        unique: list[NodeProxy] = []
        for r in results:
            if r not in seen:
                seen.add(r)
                unique.append(NodeProxy(self._chain, r))
        return unique

    def get_chain(self, only_main: bool = True) -> list[ChainProxy]:
        # Use fast index lookup instead of searching all chains
        matching: list[tuple[int, list[str]]] = self._chain._get_node_chains(
            self._node_id
        )
        if only_main:
            if not matching:
                raise ValueError(f"Node {self._node_id} is not in any chain")
            matching = [max(matching, key=lambda x: len(x[1]))]
        return [ChainProxy(self._chain, i, c) for i, c in matching]

    def get_position_in_chain(self) -> int:
        chains = self.get_chain(only_main=True)
        if not chains:
            raise ValueError(f"Node {self._node_id} is not in any chain")
        chain = chains[0]
        try:
            return chain._nodes.index(self._node_id)
        except ValueError:
            raise ValueError(f"Node {self._node_id} not found in its own chain")

    def get_component(self) -> ComponentProxy | None:
        comp_id = self._chain._node_component.get(self._node_id)
        if comp_id is None:
            return None
        return ComponentProxy(self._chain, comp_id)

    def __repr__(self) -> str:
        return f"NodeProxy({self._node_id})"


class ChainProxy:
    """Represents one directed path (chain). Created from min chain cover results."""

    def __init__(
        self, chain: ChainManager, chain_id: int, node_list: list[str]
    ) -> None:
        self._chain = chain
        self._id = chain_id
        self._nodes = node_list

    @property
    def id(self) -> int:
        return self._id

    @property
    def nodes(self) -> list[NodeProxy]:
        return [NodeProxy(self._chain, n) for n in self._nodes]

    @property
    def length(self) -> int:
        return len(self._nodes)

    @property
    def first(self) -> NodeProxy | None:
        if not self._nodes:
            return None
        return NodeProxy(self._chain, self._nodes[0])

    @property
    def last(self) -> NodeProxy | None:
        if not self._nodes:
            return None
        return NodeProxy(self._chain, self._nodes[-1])

    def get_nodes(
        self, only_top: bool = False, only_bottom: bool = False
    ) -> list[NodeProxy]:
        if only_top and only_bottom:
            raise ValueError("only_top and only_bottom cannot both be True")
        if not only_top and not only_bottom:
            return self.nodes
        result: list[NodeProxy] = []
        for n in self._nodes:
            proxy = NodeProxy(self._chain, n)
            if only_top and proxy.is_top():
                result.append(proxy)
            elif only_bottom and proxy.is_bottom():
                result.append(proxy)
        return result

    def node_position(self, node_id: str) -> int:
        try:
            return self._nodes.index(node_id)
        except ValueError:
            raise ValueError(f"Node {node_id} is not in this chain")

    def get_component(self) -> ComponentProxy | None:
        if not self._nodes:
            return None
        comp_id = self._chain._node_component.get(self._nodes[0])
        if comp_id is None:
            return None
        return ComponentProxy(self._chain, comp_id)

    def __repr__(self) -> str:
        return f"ChainProxy(id={self._id}, length={self.length})"


class ComponentProxy:
    """Represents one connected component."""

    def __init__(self, chain: ChainManager, comp_id: int) -> None:
        self._chain = chain
        self._id = comp_id

    @property
    def id(self) -> int:
        return self._id

    @property
    def nodes(self) -> list[NodeProxy]:
        return [
            NodeProxy(self._chain, n)
            for n in self._chain._component_members.get(self._id, [])
        ]

    @property
    def size(self) -> int:
        return len(self._chain._component_members.get(self._id, []))

    def get_chains(self, minimal_required: bool = True) -> list[ChainProxy]:
        all_chains = (
            self._chain.compute_min_chains()
            if minimal_required
            else self._chain.enumerate_top_chains()
        )
        comp_nodes = set(self._chain._component_members.get(self._id, []))
        matching: list[tuple[int, list[str]]] = []
        for i, c in enumerate(all_chains):
            if any(n in comp_nodes for n in c):
                matching.append((i, c))
        return [ChainProxy(self._chain, i, c) for i, c in matching]

    def __repr__(self) -> str:
        return f"ComponentProxy(id={self._id}, size={self.size})"


# ======================================================================
# CrystalGraph — main public API
# ======================================================================
nodeTuple = tuple[NodeProxy, bool]
chainTuple = tuple[ChainProxy, list[nodeTuple]]
chainDict = dict[int, chainTuple]


class CrystalGraph:
    """Main graph API. All access through get_* methods returning proxy objects."""

    def __init__(self) -> None:
        self._chain = ChainManager()
        self._images: dict[str, dict[str, Any]] = {}
        self._comparisons: list[dict[str, Any]] = []
        self._chain_map: dict[int, chainDict] | None = None
        self._rebuilding = False

    # -- Lifecycle ------------------------------------------------------

    def rebuild_from_database(
        self,
        images: list[dict[str, Any]] | None = None,
        comparisons: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._rebuilding:
            logger.debug("Already rebuilding, skipping nested call")
            return
        self._rebuilding = True
        logger.debug("Rebuilding chain from database...")
        from external_modules.step01ranking.database.images_table import (
            get_all_images,
        )
        from external_modules.step01ranking.database.comparisons_table import (
            get_all_comparisons,
        )

        if images is None:
            images = get_all_images()
        if comparisons is None:
            comparisons = get_all_comparisons()

        self._images = {img["filename"]: img for img in images}
        self._comparisons = comparisons
        self._chain._db_comparison_count = len(comparisons)
        self._chain._built_at = datetime.now(timezone.utc)

        all_filenames: set[str] = set(self._images.keys())
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

    def invalidate_cache(self) -> None:
        self._chain._built_at = None
        self._chain._db_comparison_count = 0
        self._chain_map = None

    def is_cache_stale(self) -> bool:
        if self._chain._built_at is None:
            return True
        from external_modules.step01ranking.database.comparisons_table import (
            get_total_comparisons,
        )

        return get_total_comparisons() != self._chain._db_comparison_count

    # -- Node lookups ---------------------------------------------------

    def get_node(self, node_id: str | None = None) -> NodeProxy | None:
        if node_id is None or node_id not in self._chain._all_filenames:
            return None
        return NodeProxy(self._chain, node_id)

    def get_all_nodes(
        self, only_top: bool = False, only_bottom: bool = False
    ) -> list[NodeProxy]:
        if only_top and only_bottom:
            raise ValueError("only_top and only_bottom cannot both be True")
        if only_top:
            return [NodeProxy(self._chain, n) for n in self._chain._top_nodes]
        if only_bottom:
            return [NodeProxy(self._chain, n) for n in self._chain._bottom_nodes]
        return [NodeProxy(self._chain, n) for n in self._chain._all_filenames]

    # -- Chain lookups --------------------------------------------------

    def get_chain(
        self, node_id: str | None = None, chain_id: int | None = None
    ) -> ChainProxy | None:
        if (node_id is None) == (chain_id is None):
            raise ValueError("Exactly one of node_id or chain_id is required")
        if node_id is not None:
            if node_id not in self._chain._all_filenames:
                return None
            min_chains = self._chain.compute_min_chains()
            matching = [c for c in min_chains if node_id in c]
            if not matching:
                return None
            best = max(matching, key=len)
            idx = min_chains.index(best)
            return ChainProxy(self._chain, idx, best)
        if chain_id is not None:
            min_chains = self._chain.compute_min_chains()
            if chain_id < 0 or chain_id >= len(min_chains):
                return None
            return ChainProxy(self._chain, chain_id, min_chains[chain_id])
        return None

    def get_all_chains(self) -> list[ChainProxy]:
        min_chains = self._chain.compute_min_chains()
        return [ChainProxy(self._chain, i, c) for i, c in enumerate(min_chains)]

    def get_all_sub_chains(
        self, max_chains: int = 20000000, max_depth: int = 50000
    ) -> list[ChainProxy]:
        raw = self._chain.enumerate_all_chains(
            max_chains=max_chains, max_depth=max_depth
        )
        return [ChainProxy(self._chain, i, c) for i, c in enumerate(raw)]

    # -- Component lookups ----------------------------------------------

    def get_component(
        self,
        node_id: str | None = None,
        component_id: int | None = None,
        chain_id: int | None = None,
    ) -> ComponentProxy | None:
        n_specified = sum(1 for x in (node_id, component_id, chain_id) if x is not None)
        if n_specified != 1:
            raise ValueError(
                "Exactly one of node_id, component_id, or chain_id is required"
            )
        if node_id is not None:
            cid = self._chain._node_component.get(node_id)
            if cid is None:
                return None
            return ComponentProxy(self._chain, cid)
        if component_id is not None:
            if component_id not in self._chain._component_members:
                return None
            return ComponentProxy(self._chain, component_id)
        if chain_id is not None:
            chain = self.get_chain(chain_id=chain_id)
            if chain is None or not chain._nodes:
                return None
            cid = self._chain._node_component.get(chain._nodes[0])
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
        for winner, losers in self._chain._worse_than.items():
            for loser in losers:
                key = (winner, loser)
                if key not in seen:
                    seen.add(key)
                    result.append(
                        (NodeProxy(self._chain, winner), NodeProxy(self._chain, loser))
                    )
        return result

    # -- Stats ----------------------------------------------------------

    def get_graph_stats(self) -> dict[str, Any]:
        chain_lens = self._chain._nodes_by_length.keys()
        return {
            "total_images": len(self._images) or len(self._chain._all_filenames),
            "total_comparisons": self._chain._db_comparison_count,
            "total_components": len(self._chain._component_members),
            "total_chains": self._chain.get_min_chain_count(),
            "longest_chain_depth": max(chain_lens) if chain_lens else 0,
            "top_nodes_count": len(self._chain._top_nodes),
            "bottom_nodes_count": len(self._chain._bottom_nodes),
            "built_at": (
                self._chain._built_at.isoformat() if self._chain._built_at else None
            ),
        }

    def get_cache_info(self) -> dict[str, Any]:
        return {
            "built_at": (
                self._chain._built_at.isoformat() if self._chain._built_at else None
            ),
            "comparison_count_at_build": self._chain._db_comparison_count,
            "is_stale": self.is_cache_stale(),
        }

    # -- Delegated helpers (used externally) ----------------------------

    def is_redundant(
        self,
        u: str,
        v: str,
        skip_edges: set[tuple[str, str]] | None = None,
    ) -> bool:
        return self._chain.is_redundant(u, v, skip_edges)

    def are_in_same_path(self, img1: str, img2: str) -> bool:
        return self._chain.are_in_same_path(img1, img2)

    def get_collapsable_pairs(self) -> list[tuple[str, str]]:
        return self._chain.get_collapsable_pairs()

    def get_chains_map(self) -> dict[int, chainDict]:
        if self._chain_map is not None:
            return self._chain_map

        chains: list[ChainProxy] = crystal_graph.get_all_chains()
        chain_map: dict[int, list[ChainProxy]] = {}
        with tqdm(total=len(chains), desc="Mapping chains by length") as pbar:
            for c in chains:
                if c.length not in chain_map:
                    chain_map[c.length] = []
                chain_map[c.length].append(c)
                pbar.update(1)

        # sort map by length
        chain_map = dict(sorted(chain_map.items(), key=lambda item: item[0]))

        all_nodes: list[NodeProxy] = crystal_graph.get_all_nodes()

        main_chains: dict[int, list[str]] = {}
        with tqdm(all_nodes, desc="Processing nodes") as pbar:
            for node in all_nodes:
                chain_id: int = node.get_chain()[0].id
                if chain_id not in main_chains:
                    main_chains[chain_id] = []
                main_chains[chain_id].append(node.filename)
                pbar.update(1)

        final_map: dict[int, chainDict] = {}
        errors: list[int] = []
        # dict [chain_length, dict[chain_id, list[tuple[node, is_main_node]]]]
        with tqdm(total=len(chains), desc="Building final map") as pbar:
            for lengths in chain_map:
                for chain in chain_map[lengths]:
                    if chain.id not in main_chains:
                        errors.append(chain.id)
                        # print(f"Warning: Chain ID {chain.id} not found in main_chains")
                        continue

                    if chain.length not in final_map:
                        final_map[chain.length] = {}

                    local_main_chains: list[str] = main_chains[chain.id]

                    final_chain: list[nodeTuple] = []
                    nodes_in_chain: list[NodeProxy] = chain.get_nodes()

                    for node in nodes_in_chain:
                        is_main_node: bool = node.filename in local_main_chains
                        final_chain.append((node, is_main_node))

                    final_map[chain.length][chain.id] = (chain, final_chain)
                    pbar.update(1)
        self._chain_map = final_map
        logger.debug(
            f"Chain mapping completed with {len(errors)}"
            # f"missing chains: {errors}"
        )
        return self._chain_map

    def get_duplicate_comparison_ids(self) -> list[int]:
        """Return a list of comparison record IDs that are duplicates.

        For each unique (canonical_pair, winner) combination, keeps the most recent
        record (by timestamp) and marks older duplicates for deletion.

        Handles both:
        - Same-direction duplicates: (a,b,winner=a) and (b,a,winner=a)
        - Exact duplicates: identical records appearing multiple times
        """
        from collections import defaultdict

        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for comp in self._comparisons:
            a = comp.get("filename_a", "")
            b = comp.get("filename_b", "")
            winner = comp.get("winner", "")
            canon_a, canon_b = tuple(sorted((a, b)))
            key = (canon_a, canon_b, winner)
            groups[key].append(comp)

        ids_to_delete: list[int] = []
        for key, comps in groups.items():
            if len(comps) > 1:
                sorted_comps = sorted(
                    comps, key=lambda c: c.get("timestamp", ""), reverse=True
                )
                for comp in sorted_comps[1:]:
                    comp_id = comp.get("id")
                    if comp_id is not None:
                        ids_to_delete.append(comp_id)

        return ids_to_delete

    def get_redundant_edges(self) -> set[tuple[str, str]]:
        """Return a set of directed edges (winner, loser) that are considered redundant.

        Redundancy rules implemented:
        - Self-links (u -> u) are redundant.
        - Transitively redundant edges: direct edge u->v is redundant if there exists
          an alternative path u -> ... -> v (length >= 2) ignoring the direct edge.
        - Contradictory double-links (a->b and b->a): keep the most recent winner
          (based on comparison timestamps) and mark the opposite edge as redundant.
        """
        from collections import defaultdict

        redundant_edges: set[tuple[str, str]] = set()

        # 1. Detect self-links (image compared to itself)
        for u, losers in self._chain._worse_than.items():
            for v in losers:
                if u == v:
                    # logger.debug(f"Found self-link: ({u} -> {v})")
                    redundant_edges.add((u, v))

        # 3. Detect double links (a->b and b->a) and mark the older as redundant
        # Note: DB stores pairs in canonical (sorted) order, so both contradictory
        # records are under the same key. Check within each group for conflicting winners.
        pair_comparisons: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
            list
        )
        for comp in self._comparisons:
            key = (comp.get("filename_a"), comp.get("filename_b"))
            pair_comparisons[key].append(comp)

        for (ca, cb), comps in pair_comparisons.items():
            winners = set(c.get("winner") for c in comps if c.get("winner") is not None)
            if ca in winners and cb in winners:
                edge_ab = (ca, cb)
                edge_ba = (cb, ca)
                if edge_ab in redundant_edges or edge_ba in redundant_edges:
                    continue
                ts_a = max(
                    (c["timestamp"] for c in comps if c.get("winner") == ca),
                )
                ts_b = max(
                    (c["timestamp"] for c in comps if c.get("winner") == cb),
                )
                if ts_a >= ts_b:
                    edge_to_remove = edge_ba
                else:
                    edge_to_remove = edge_ab

                # logger.debug(
                #     f"Found contradictory double-link between {ca} and {cb}. Keeping winner with timestamp {max(ts_a, ts_b)} and marking edge {edge_to_remove} as redundant."
                # )
                redundant_edges.add(edge_to_remove)

        # 2. Detect transitively redundant directed edges via chain.is_redundant
        for u, losers in self._chain._worse_than.items():
            for v in losers:
                if u != v and self._chain.is_redundant(u, v):
                    # logger.debug(f"Found transitively redundant edge: ({u} -> {v})")
                    redundant_edges.add((u, v))

        return redundant_edges


crystal_graph = CrystalGraph()
crystal_graph.rebuild_from_database()
