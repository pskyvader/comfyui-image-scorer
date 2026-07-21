from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any
import time
from tqdm import tqdm

from ..logger import get_logger

logger = get_logger(__name__)


# ======================================================================
# Tiny helpers (pure, no self)
# ======================================================================


def parse_comparison(
    comp: dict[str, Any],
) -> tuple[str, str, str, str]:
    a: str = comp["filename_a"]
    b: str = comp["filename_b"]
    winner: str = comp["winner"]
    loser: str = b if winner == a else a
    return a, b, winner, loser


def add_directed_edge(
    better_than: defaultdict[str, list[str]],
    worse_than: defaultdict[str, list[str]],
    winner: str,
    loser: str,
) -> None:
    better_than[loser].append(winner)
    worse_than[winner].append(loser)


def add_undirected_edge(
    adjacency: defaultdict[str, set[str]],
    filenames: set[str],
    a: str,
    b: str,
) -> None:
    adjacency[a].add(b)
    adjacency[b].add(a)
    filenames.add(a)
    filenames.add(b)


def process_one_comparison(
    comp: dict[str, Any],
    better_than: defaultdict[str, list[str]],
    worse_than: defaultdict[str, list[str]],
    adjacency: defaultdict[str, set[str]],
    filenames: set[str],
) -> None:
    a, b, winner, loser = parse_comparison(comp)
    add_directed_edge(better_than, worse_than, winner, loser)
    add_undirected_edge(adjacency, filenames, a, b)


# ------------------------------------------------------------------
# Top / bottom detection
# ------------------------------------------------------------------


def has_no_predecessors(
    node: str,
    better_than: defaultdict[str, list[str]],
) -> bool:
    return not better_than[node]


def has_no_successors(
    node: str,
    worse_than: defaultdict[str, list[str]],
) -> bool:
    return not worse_than[node]


def find_top_nodes(
    all_filenames: set[str],
    better_than: defaultdict[str, list[str]],
) -> set[str]:
    return {n for n in all_filenames if has_no_predecessors(n, better_than)}


def find_bottom_nodes(
    all_filenames: set[str],
    worse_than: defaultdict[str, list[str]],
) -> set[str]:
    return {n for n in all_filenames if has_no_successors(n, worse_than)}


# ------------------------------------------------------------------
# Components (BFS)
# ------------------------------------------------------------------


def bfs_one_component(
    start: str,
    adjacency: defaultdict[str, set[str]],
    visited: set[str],
) -> list[str]:
    queue: deque[str] = deque([start])
    visited.add(start)
    members: list[str] = []
    while queue:
        current: str = queue.popleft()
        members.append(current)
        neighbors = adjacency.get(current, set())
        # with tqdm(total=len(neighbors), desc="BFS neighbors", unit="edge", leave=False) as pbar:
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                # pbar.update(1)
    return members


def index_component(
    members: list[str],
    comp_id: int,
    node_component: dict[str, int],
    component_members: dict[int, list[str]],
) -> None:
    component_members[comp_id] = members
    with tqdm(
        total=len(members), desc="Indexing component", unit="node", leave=False
    ) as pbar:
        for member in members:
            node_component[member] = comp_id
            pbar.update(1)


def build_components(
    all_filenames: set[str],
    adjacency: defaultdict[str, set[str]],
) -> tuple[dict[str, int], dict[int, list[str]]]:
    visited: set[str] = set()
    node_component: dict[str, int] = {}
    component_members: dict[int, list[str]] = {}
    comp_id: int = 0

    for filename in all_filenames:
        if filename in visited:
            continue
        members: list[str] = bfs_one_component(filename, adjacency, visited)
        index_component(members, comp_id, node_component, component_members)
        comp_id += 1

    return node_component, component_members


# ------------------------------------------------------------------
# Reachability helpers
# ------------------------------------------------------------------


def same_component(
    u: str,
    v: str,
    node_component: dict[str, int],
) -> bool | None:
    cu = node_component.get(u)
    cv = node_component.get(v)
    if cu is not None and cv is not None:
        return cu == cv
    return None


def find_common_chain_id(
    chains_u: dict[int, bool],
    chains_v: dict[int, bool],
) -> int | None:
    with tqdm(
        total=len(chains_u), desc="Finding common chain", unit="cid", leave=False
    ) as pbar:
        for cid in chains_u:
            if cid in chains_v:
                return cid
            pbar.update(1)
    return None


# ======================================================================
# ChainManager class
# ======================================================================


class ChainManager:
    def __init__(self) -> None:
        self._better_than: defaultdict[str, list[str]] = defaultdict(list)
        self._worse_than: defaultdict[str, list[str]] = defaultdict(list)
        self._adjacency: defaultdict[str, set[str]] = defaultdict(set)

        self._top_nodes: set[str] = set()
        self._bottom_nodes: set[str] = set()

        self._node_component: dict[str, int] = {}
        self._component_members: dict[int, list[str]] = {}

        self._all_filenames: set[str] = set()
        self._node_pos: dict[str, int] = {}
        self._chain_end_at: dict[str, set[int]] = {}
        self._built_at: datetime | None = None
        self._db_comparison_count: int = 0

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
        return list(self._top_nodes)

    def get_bottom_nodes(self) -> list[str]:
        return list(self._bottom_nodes)

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
        _start: float = time.perf_counter()

        self._reset_adjacency()
        graph_filenames: set[str] = self._build_from_comparisons(comparisons)
        self._all_filenames = (
            all_filenames if all_filenames is not None else graph_filenames
        )
        self._identify_top_bottom()
        self._build_components()
        self._build_chains()
        logger.info("build complete", start_timer=_start)

    def _reset_adjacency(self) -> None:
        self._better_than.clear()
        self._worse_than.clear()
        self._adjacency.clear()

    def _build_from_comparisons(self, comparisons: list[dict[str, Any]]) -> set[str]:
        graph_filenames: set[str] = set()
        with tqdm(
            total=len(comparisons), desc="Building graph from comparisons", unit="comp"
        ) as pbar:
            for comp in comparisons:
                process_one_comparison(
                    comp,
                    self._better_than,
                    self._worse_than,
                    self._adjacency,
                    graph_filenames,
                )
                pbar.update(1)
        return graph_filenames

    # ==================================================================
    # Incremental update (single comparison, no DB)
    # ==================================================================

    def apply_comparison(self, winner: str, loser: str) -> None:
        self._all_filenames.update([winner, loser])

        add_directed_edge(self._better_than, self._worse_than, winner, loser)
        add_undirected_edge(self._adjacency, set(), winner, loser)

        self._update_top_bottom_for_edge(winner, loser)
        self._merge_node_components(winner, loser)

        self._db_comparison_count += 1
        self._built_at = datetime.now(timezone.utc)

    def _remove_from_bottom_if_not_anymore(self, winner: str) -> None:
        if winner in self._bottom_nodes and self._worse_than[winner]:
            self._bottom_nodes.discard(winner)

    def _remove_from_top_if_not_anymore(self, loser: str) -> None:
        if loser in self._top_nodes and self._better_than[loser]:
            self._top_nodes.discard(loser)

    def _add_to_bottom_if_needed(self, loser: str) -> None:
        if not self._worse_than[loser]:
            self._bottom_nodes.add(loser)

    def _add_to_top_if_needed(self, winner: str) -> None:
        if not self._better_than[winner]:
            self._top_nodes.add(winner)

    def _update_top_bottom_for_edge(self, winner: str, loser: str) -> None:
        self._remove_from_bottom_if_not_anymore(winner)
        self._remove_from_top_if_not_anymore(loser)
        self._add_to_bottom_if_needed(loser)
        self._add_to_top_if_needed(winner)

    def _component_of(self, node: str) -> int | None:
        return self._node_component.get(node)

    def _both_have_components_and_different(
        self, cw: int | None, cl: int | None
    ) -> bool:
        return cw is not None and cl is not None and cw != cl

    def _neither_has_component(self, cw: int | None, cl: int | None) -> bool:
        return cw is None and cl is None

    def _winner_lacks_component(self, cw: int | None, cl: int | None) -> bool:
        return cw is None and cl is not None

    def _loser_lacks_component(self, cw: int | None, cl: int | None) -> bool:
        return cl is None and cw is not None

    def _create_new_component(self, winner: str, loser: str) -> None:
        new_id: int = max(self._component_members.keys(), default=-1) + 1
        self._component_members[new_id] = [winner, loser]
        self._node_component[winner] = new_id
        self._node_component[loser] = new_id

    def _add_winner_to_loser_component(self, winner: str, cl: int) -> None:
        self._node_component[winner] = cl
        self._component_members[cl].append(winner)

    def _add_loser_to_winner_component(self, loser: str, cw: int) -> None:
        self._node_component[loser] = cw
        self._component_members[cw].append(loser)

    def _merge_node_components(self, winner: str, loser: str) -> None:
        cw: int | None = self._component_of(winner)
        cl: int | None = self._component_of(loser)

        if self._both_have_components_and_different(cw, cl):
            self._merge_components(cw, cl)
        elif self._neither_has_component(cw, cl):
            self._create_new_component(winner, loser)
        elif self._winner_lacks_component(cw, cl):
            self._add_winner_to_loser_component(winner, cl)
        elif self._loser_lacks_component(cw, cl):
            self._add_loser_to_winner_component(loser, cw)

    # ------------------------------------------------------------------
    # Component merging
    # ------------------------------------------------------------------

    def _ensure_larger_component_kept(
        self, keep_id: int, remove_id: int
    ) -> tuple[int, int]:
        if len(self._component_members.get(keep_id, [])) < len(
            self._component_members.get(remove_id, [])
        ):
            return remove_id, keep_id
        return keep_id, remove_id

    def _reassign_nodes(self, remove_id: int, keep_id: int) -> None:
        with tqdm(
            total=len(self._component_members.get(remove_id, [])),
            desc="Reassigning nodes",
            unit="node",
            leave=False,
        ) as pbar:
            for node in self._component_members.get(remove_id, []):
                self._node_component[node] = keep_id
                pbar.update(1)

    def _absorb_removed_component(self, keep_id: int, remove_id: int) -> None:
        self._component_members[keep_id].extend(
            self._component_members.pop(remove_id, [])
        )

    def _merge_components(self, keep_id: int, remove_id: int) -> None:
        keep_id, remove_id = self._ensure_larger_component_kept(keep_id, remove_id)
        with tqdm(
            self._component_members.get(remove_id, []),
            desc="Merging components",
            unit="node",
        ) as pbar:
            for node in pbar:
                self._node_component[node] = keep_id
        self._absorb_removed_component(keep_id, remove_id)

    # ==================================================================
    # Internal build helpers
    # ==================================================================

    def _identify_top_bottom(self) -> None:
        self._top_nodes = find_top_nodes(self._all_filenames, self._better_than)
        self._bottom_nodes = find_bottom_nodes(self._all_filenames, self._worse_than)

    def _build_components(self) -> None:
        self._node_component, self._component_members = build_components(
            self._all_filenames,
            self._adjacency,
        )

    # ------------------------------------------------------------------
    # Chain building
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_path(path: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        # with tqdm(
        #     total=len(path), desc="Deduping path", unit="node", leave=False
        # ) as pbar:
        for x in path:
            if x in seen:
                break
            seen.add(x)
            result.append(x)
            # pbar.update(1)
        return result

    def _build_chains(self) -> None:
        _start: float = time.perf_counter()
        self._node_main_chain = {}
        self._chains = {}
        self._node_chains = {}

        # Topological order for DP
        in_degree: dict[str, int] = {}
        with tqdm(
            total=len(self._all_filenames), desc="Initializing in-degrees", unit="node"
        ) as pbar:
            for n in self._all_filenames:
                in_degree[n] = 0
                pbar.update(1)
        with tqdm(
            total=sum(len(v) for v in self._worse_than.values()),
            desc="Building in-degrees",
            unit="edges",
        ) as pbar:
            for children in self._worse_than.values():
                for child in children:
                    in_degree[child] = in_degree.get(child, 0) + 1
                    pbar.update(1)

        order: list[str] = []
        queue: deque[str] = deque(
            n for n in self._all_filenames if in_degree.get(n, 0) == 0
        )
        remaining: set[str] = set(self._all_filenames)
        with tqdm(
            total=len(self._all_filenames), desc="Topological sort", unit="node"
        ) as pbar:
            while remaining:
                while queue:
                    node: str = queue.popleft()
                    if node not in remaining:
                        continue
                    remaining.discard(node)
                    order.append(node)
                    pbar.update(1)
                    children = self._worse_than.get(node, [])
                    # with tqdm(
                    #     total=len(children),
                    #     desc="Processing children",
                    #     unit="edge",
                    #     leave=False,
                    # ) as cbar:
                    for child in children:
                        in_degree[child] -= 1
                        if in_degree[child] == 0:
                            queue.append(child)
                            # cbar.update(1)
                if not remaining:
                    break
                pick: str = sorted(remaining)[0]
                queue.append(pick)

        # Forward DP (longest path to a bottom, prefer bottom reachable)
        fwd: dict[str, list[str]] = {}
        with tqdm(
            total=len(self._all_filenames), desc="Initializing forward DP", unit="node"
        ) as pbar:
            for n in self._all_filenames:
                fwd[n] = [n]
                pbar.update(1)
        changed: bool = True
        # pass_num: int = 0

        with tqdm(
            # total=len(order),
            desc=f"Forward DP pass",
            unit="node",
            # leave=False,
        ) as pbar:
            while changed:
                pbar.update(1)
                # pass_num += 1
                changed = False
                for n in reversed(order):
                    losers: list[str] = self._worse_than.get(n, [])
                    if not losers:
                        fwd[n] = [n]
                        continue
                    best: list[str] = []
                    best_to_bottom: list[str] = []
                    # with tqdm(
                    #     total=len(losers),
                    #     desc="Processing losers",
                    #     unit="loser",
                    #     leave=False,
                    # ) as pbar2:
                    for l in losers:
                        p = fwd.get(l, [l])
                        if n in p:
                            p = [l]
                        if len(p) > len(best):
                            best = p
                        if self.is_bottom(p[-1]) and len(p) > len(best_to_bottom):
                            best_to_bottom = p
                            # pbar2.update(1)
                    new: list[str] = [n] + (best_to_bottom if best_to_bottom else best)
                    if len(new) > len(fwd[n]):
                        fwd[n] = new
                        changed = True
                    # pbar.update(1)

        # Backward DP (longest path from a top, prefer top reachable)
        bwd: dict[str, list[str]] = {}
        with tqdm(
            total=len(self._all_filenames), desc="Initializing backward DP", unit="node"
        ) as pbar:
            for n in self._all_filenames:
                bwd[n] = [n]
                pbar.update(1)
        changed = True
        # pass_num = 0
        with tqdm(
            # total=len(order),
            desc=f"Backward DP pass",
            unit="node",
            # leave=False,
        ) as pbar:
            while changed:
                pbar.update(1)

                # pass_num += 1
                changed = False

                for n in order:
                    beaters: list[str] = self._better_than.get(n, [])
                    if not beaters:
                        bwd[n] = [n]
                        # pbar.update(1)
                        continue
                    best: list[str] = []
                    best_to_top: list[str] = []
                    # with tqdm(
                    #     total=len(beaters),
                    #     desc="Processing beaters",
                    #     unit="beater",
                    #     leave=False,
                    # ) as pbar2:
                    for b in beaters:
                        p = bwd.get(b, [b])
                        if n in p:
                            p = [b]
                        if len(p) > len(best):
                            best = p
                        if self.is_top(p[0]) and len(p) > len(best_to_top):
                            best_to_top = p
                            # pbar2.update(1)
                    new = (best_to_top if best_to_top else best) + [n]
                    if len(new) > len(bwd[n]):
                        bwd[n] = new
                        changed = True
                    # pbar.update(1)

        # Build unique chains and assign main chains
        seen: dict[tuple[str, ...], int] = {}
        next_id: int = 0
        with tqdm(
            total=len(self._all_filenames), desc="Building main chains", unit="node"
        ) as pbar:
            for n in self._all_filenames:
                fp: list[str] = self._dedup_path(fwd[n])
                bp: list[str] = self._dedup_path(bwd[n])
                if len(bp) > 1:
                    prefix: set[str] = set(bp[:-1])
                    full: list[str] = bp[:-1] + [x for x in fp if x not in prefix]
                else:
                    full = fp
                key: tuple[str, ...] = tuple(full)
                if key not in seen:
                    seen[key] = next_id
                    self._chains[next_id] = full
                    next_id += 1
                cid: int = seen[key]
                self._node_main_chain[n] = (cid, full)
                self._node_chains[n] = {cid: True}
                self._node_to_chains.setdefault(n, []).append((cid, full))
                pbar.update(1)

        with tqdm(
            total=len(self._chains), desc="Setting common chains", unit="chain"
        ) as pbar:
            for cid, chain in self._chains.items():
                self._common_chains[cid] = (chain, True)
                pbar.update(1)

        logger.info(
            f"Built {len(self._chains)} main chains",
            start_timer=_start,
        )

    # ==================================================================
    # Chain accessors
    # ==================================================================

    def get_chains(self) -> dict[int, list[str]]:
        return self._chains

    def get_node_chains(self, node_id: str) -> list[tuple[int, list[str]]]:
        return self._node_to_chains.get(node_id, [])

    def get_node_main_chain(self, node_id: str) -> tuple[int, list[str]] | None:
        return self._node_main_chain.get(node_id)

    def get_min_chain_count(self) -> int:
        return len(self._chains)

    # ==================================================================
    # Reachability
    # ==================================================================

    def _quick_reject(self, start: str, end: str) -> str | None:
        if start not in self._all_filenames or end not in self._all_filenames:
            return "missing node"
        same = same_component(start, end, self._node_component)
        if same is False:
            return "different component"
        if not self._worse_than.get(start):
            return "start has no outgoing"
        if not self._better_than.get(end):
            return "end has no incoming"
        return None

    def _bfs_search(
        self,
        start: str,
        end: str,
        skip_edges: set[tuple[str, str]] | None,
        max_depth: int,
    ) -> bool:
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            children = self._worse_than.get(current, [])
            with tqdm(
                total=len(children), desc="BFS search", unit="edge", leave=False
            ) as pbar:
                for w in children:
                    if skip_edges and (current, w) in skip_edges:
                        pbar.update(1)
                        continue
                    if w == end:
                        return True
                    if w not in visited:
                        visited.add(w)
                        queue.append((w, depth + 1))
                    pbar.update(1)
        return False

    def _can_reach(
        self,
        start: str,
        end: str,
        skip_edges: set[tuple[str, str]] | None = None,
        max_depth: int = 3,
    ) -> bool:
        _start = time.perf_counter()
        reject: str | None = self._quick_reject(start, end)
        if reject is not None:
            return False
        if start == end:
            return True
        if not skip_edges:
            same_chain, _ = self._check_same_chain(start, end)
            if same_chain:
                logger.debug(f"can reach (same chain)", start_timer=_start)
                return True
        found: bool = self._bfs_search(start, end, skip_edges, max_depth)
        if found:
            logger.debug(f"can reach", start_timer=_start)
        else:
            logger.debug(f"cannot reach", start_timer=_start)
        return found

    def _check_same_chain(self, u: str, v: str) -> tuple[bool, bool]:
        chains_u = self._node_chains.get(u)
        chains_v = self._node_chains.get(v)
        if not chains_u or not chains_v:
            return (False, False)
        common_id: int | None = find_common_chain_id(chains_u, chains_v)
        if common_id is None:
            return (False, False)
        common_chain: list[str] = self._common_chains[common_id][0]
        u_position = common_chain.index(u)
        v_position = common_chain.index(v)
        return (True, u_position < v_position)
