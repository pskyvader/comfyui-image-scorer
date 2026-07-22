from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any
import time
from tqdm import tqdm

from ..logger import get_logger, ModuleLogger
from ..config import config

logger: ModuleLogger = get_logger(__name__)


# ======================================================================
# Tiny helpers (pure, no self)
# ======================================================================


def parse_comparison(
    comp: dict[str, Any],
) -> tuple[str, str, str, str]:
    filename_a: str = comp["filename_a"]
    filename_b: str = comp["filename_b"]
    winner: str = comp["winner"]
    loser: str = filename_b if winner == filename_a else filename_a
    return filename_a, filename_b, winner, loser


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
    filename_a: str,
    filename_b: str,
) -> None:
    adjacency[filename_a].add(filename_b)
    adjacency[filename_b].add(filename_a)
    filenames.add(filename_a)
    filenames.add(filename_b)


def process_one_comparison(
    comp: dict[str, Any],
    better_than: defaultdict[str, list[str]],
    worse_than: defaultdict[str, list[str]],
    adjacency: defaultdict[str, set[str]],
    filenames: set[str],
) -> None:
    filename_a, filename_b, winner, loser = parse_comparison(comp)
    add_directed_edge(better_than, worse_than, winner, loser)
    add_undirected_edge(adjacency, filenames, filename_a, filename_b)


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
        total=len(members),
        desc="Indexing component",
        unit="node",
        leave=False,
        delay=3.0,
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
    comp_u = node_component.get(u)
    comp_v = node_component.get(v)
    if comp_u is not None and comp_v is not None:
        return comp_u == comp_v
    return None


def find_common_chain_id(
    node_chains: dict[int, bool],
    other_chains: dict[int, bool],
) -> int | None:
    with tqdm(
        total=len(node_chains),
        desc="Finding common chain",
        unit="chain",
        leave=False,
        delay=3.0,
    ) as pbar:
        for chain_id in node_chains:
            if chain_id in other_chains:
                return chain_id
            pbar.update(1)
    return None


# ------------------------------------------------------------------
# Tarjan SCC (cycle condensation)
# ------------------------------------------------------------------


def tarjan_scc(
    nodes: set[str],
    successors: defaultdict[str, list[str]],
) -> tuple[dict[str, int], dict[int, list[str]]]:
    index_counter: int = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    scc_id_of: dict[str, int] = {}
    scc_members: dict[int, list[str]] = {}
    scc_count: int = 0

    def strongconnect(v: str) -> None:
        nonlocal index_counter, scc_count
        indices[v] = index_counter
        lowlinks[v] = index_counter
        index_counter += 1
        stack.append(v)
        on_stack.add(v)

        for w in successors.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            component: list[str] = []
            while True:
                w: str = stack.pop()
                on_stack.discard(w)
                component.append(w)
                scc_id_of[w] = scc_count
                if w == v:
                    break
            scc_members[scc_count] = component
            scc_count += 1

    for v in sorted(nodes):
        if v not in indices:
            strongconnect(v)

    return scc_id_of, scc_members


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
            total=len(comparisons),
            desc="Building graph from comparisons",
            unit="comp",
            delay=3.0,
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
            delay=3.0,
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
            delay=3.0,
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

        # 1. Find SCCs (condense cycles into single nodes)
        scc_id_of, scc_members = tarjan_scc(
            self._all_filenames,
            self._worse_than,
        )

        # 2. Build condensed DAG (edges between different SCCs)
        scc_successors: dict[int, set[int]] = {}
        for scc_id in scc_members:
            scc_successors[scc_id] = set()
        for n in self._all_filenames:
            n_scc = scc_id_of[n]
            for v in self._worse_than.get(n, []):
                v_scc = scc_id_of[v]
                if n_scc != v_scc:
                    scc_successors[n_scc].add(v_scc)

        scc_predecessors: dict[int, set[int]] = {}
        for scc_id in scc_members:
            scc_predecessors[scc_id] = set()
        for n in self._all_filenames:
            n_scc = scc_id_of[n]
            for parent in self._better_than.get(n, []):
                p_scc = scc_id_of[parent]
                if n_scc != p_scc:
                    scc_predecessors[n_scc].add(p_scc)

        # 3. Topological sort of SCC DAG
        scc_in_degree: dict[int, int] = {
            scc: len(scc_predecessors.get(scc, [])) for scc in scc_members
        }
        scc_order: list[int] = []
        queue: deque[int] = deque(s for s, deg in scc_in_degree.items() if deg == 0)
        remaining_sccs: set[int] = set(scc_members.keys())
        while remaining_sccs:
            while queue:
                scc = queue.popleft()
                if scc not in remaining_sccs:
                    continue
                remaining_sccs.discard(scc)
                scc_order.append(scc)
                for succ in scc_successors.get(scc, []):
                    scc_in_degree[succ] -= 1
                    if scc_in_degree[succ] == 0:
                        queue.append(succ)
            if not remaining_sccs:
                break
            pick = min(remaining_sccs)
            queue.append(pick)

        # 4. Forward DP (single pass on SCC DAG, reverse topo order)
        downward_chains: dict[str, list[str]] = {}
        for n in self._all_filenames:
            downward_chains[n] = [n]

        for scc_id in reversed(scc_order):
            members: list[str] = scc_members[scc_id]

            # Phase A: incorporate edges to successor SCCs (already processed)
            for n in members:
                best: list[str] = [n]
                best_to_bottom: list[str] = []
                for loser in self._worse_than.get(n, []):
                    if scc_id_of[loser] == scc_id:
                        continue
                    path = downward_chains[loser]
                    candidate = [n] + path
                    if len(candidate) > len(best):
                        best = candidate
                    if self.is_bottom(candidate[-1]) and len(candidate) > len(
                        best_to_bottom
                    ):
                        best_to_bottom = candidate
                downward_chains[n] = best_to_bottom if best_to_bottom else best

            # Phase B: propagate external suffixes backwards through SCC
            internal_pred: dict[str, list[str]] = {}
            for n in members:
                for loser in self._worse_than.get(n, []):
                    if scc_id_of[loser] == scc_id:
                        internal_pred.setdefault(loser, []).append(n)
            q: deque[str] = deque(n for n in members if len(downward_chains[n]) > 1)
            while q:
                src = q.popleft()
                src_path = downward_chains[src]
                for pred in internal_pred.get(src, []):
                    if pred in src_path:
                        cand = [pred, src]
                    else:
                        cand = [pred] + src_path
                    cur = downward_chains[pred]
                    new_bttm = self.is_bottom(cand[-1])
                    old_bttm = self.is_bottom(cur[-1])
                    if new_bttm and not old_bttm:
                        downward_chains[pred] = cand
                        q.append(pred)
                    elif new_bttm == old_bttm and len(cand) > len(cur):
                        downward_chains[pred] = cand
                        q.append(pred)

        # 5. Backward DP (single pass on SCC DAG, forward topo order)
        upward_chains: dict[str, list[str]] = {}
        for n in self._all_filenames:
            upward_chains[n] = [n]

        for scc_id in scc_order:
            members = scc_members[scc_id]

            # Phase A: incorporate edges from predecessor SCCs (already processed)
            for n in members:
                best = [n]
                best_to_top: list[str] = []
                for beater in self._better_than.get(n, []):
                    if scc_id_of[beater] == scc_id:
                        continue
                    path = upward_chains[beater]
                    candidate = path + [n]
                    if len(candidate) > len(best):
                        best = candidate
                    if self.is_top(candidate[0]) and len(candidate) > len(best_to_top):
                        best_to_top = candidate
                upward_chains[n] = best_to_top if best_to_top else best

            # Phase B: propagate external prefixes forward through SCC
            internal_succ: dict[str, list[str]] = {}
            for n in members:
                for beater in self._better_than.get(n, []):
                    if scc_id_of[beater] == scc_id:
                        internal_succ.setdefault(beater, []).append(n)
            q = deque(n for n in members if len(upward_chains[n]) > 1)
            while q:
                src = q.popleft()
                src_path = upward_chains[src]
                for succ in internal_succ.get(src, []):
                    if succ in src_path:
                        cand = [src, succ]
                    else:
                        cand = src_path + [succ]
                    cur = upward_chains[succ]
                    new_tp = self.is_top(cand[0])
                    old_tp = self.is_top(cur[0])
                    if new_tp and not old_tp:
                        upward_chains[succ] = cand
                        q.append(succ)
                    elif new_tp == old_tp and len(cand) > len(cur):
                        upward_chains[succ] = cand
                        q.append(succ)

        # Build unique chains and assign main chains
        seen: dict[tuple[str, ...], int] = {}
        next_id: int = 0
        with tqdm(
            total=len(self._all_filenames),
            desc="Building main chains",
            unit="node",
            delay=3.0,
        ) as pbar:
            for n in self._all_filenames:
                down_chain: list[str] = self._dedup_path(downward_chains[n])
                up_chain: list[str] = self._dedup_path(upward_chains[n])
                if len(up_chain) > 1:
                    prefix: set[str] = set(up_chain[:-1])
                    full: list[str] = up_chain[:-1] + [
                        x for x in down_chain if x not in prefix
                    ]
                else:
                    full = down_chain
                chain_key: tuple[str, ...] = tuple(full)
                if chain_key not in seen:
                    seen[chain_key] = next_id
                    self._chains[next_id] = full
                    next_id += 1
                chain_id: int = seen[chain_key]
                self._node_main_chain[n] = (chain_id, full)
                self._node_chains[n] = {chain_id: True}
                self._node_to_chains.setdefault(n, []).append((chain_id, full))
                pbar.update(1)

        with tqdm(
            total=len(self._chains),
            desc="Setting common chains",
            unit="chain",
            delay=3.0,
        ) as pbar:
            for chain_id, chain in self._chains.items():
                self._common_chains[chain_id] = (chain, True)
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
        # skip_edges: set[tuple[str, str]] | None,
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
                total=len(children),
                desc="BFS search",
                unit="edge",
                leave=False,
                delay=3.0,
            ) as pbar:
                for w in children:
                    # if skip_edges and (current, w) in skip_edges:
                    #     pbar.update(1)
                    #     continue
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
        # skip_edges: set[tuple[str, str]] | None = None,
        # max_depth: int = 10,
    ) -> bool:
        _start = time.perf_counter()
        max_depth = config["ranking"]["transitive_depth"]
        reject: str | None = self._quick_reject(start, end)
        if reject is not None:
            return False
        if start == end:
            return True
        # if not skip_edges:
        same_chain, _ = self._check_same_chain(start, end)
        if same_chain:
            logger.debug(f"can reach (same chain)", start_timer=_start)
            return True
        found: bool = self._bfs_search(start, end, max_depth)
        # if found:
        #     logger.debug(f"can reach", start_timer=_start)
        # else:
        #     logger.debug(f"cannot reach", start_timer=_start)
        return found

    def _check_same_chain(self, u: str, v: str) -> tuple[bool, bool]:
        node_chains = self._node_chains.get(u)
        other_chains = self._node_chains.get(v)
        if not node_chains or not other_chains:
            return (False, False)
        common_id: int | None = find_common_chain_id(node_chains, other_chains)
        if common_id is None:
            return (False, False)
        common_chain: list[str] = self._common_chains[common_id][0]
        u_position = common_chain.index(u)
        v_position = common_chain.index(v)
        return (True, u_position < v_position)
