from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any


class ChainManager:
    """Single source of truth for all chain (directed path) computation in the comparison graph.

    A chain is a directed path along winner→loser edges in the tournament graph.
    This class provides:
    - Building adjacency from comparison records
    - Chain length = longest chain through each node (DP-based)
    - All-chains enumeration (exhaustive DFS, used for score redistribution)
    - Top-to-leaf chains enumeration
    - Connected component detection (via undirected graph)
    - Reachability / same-path / redundancy queries
    """

    def __init__(self) -> None:
        self.better: defaultdict[str, list[str]] = defaultdict(list)
        self.worse: defaultdict[str, list[str]] = defaultdict(list)
        self.chain_length: dict[str, int] = {}
        self.nodes_by_chain_length: dict[int, list[str]] = defaultdict(list)
        self.top_nodes: list[str] = []
        self.bottom_nodes: list[str] = []
        self.undirected_graph: defaultdict[str, set[str]] = defaultdict(set)
        self.component_by_filename: dict[str, int] = {}
        self.chain_members_by_component: dict[int, list[str]] = {}
        self._all_filenames: set[str] = set()

    def build(
        self,
        comparisons: list[dict[str, Any]],
        all_filenames: set[str] | None = None,
    ) -> None:
        """Build adjacency and compute all chain structures from comparisons.

        Args:
            comparisons: List of comparison dicts with 'filename_a', 'filename_b', 'winner'.
            all_filenames: Optional set of ALL filenames (including isolated images).
                           If omitted, only filenames appearing in comparisons are used.
        """
        self.better.clear()
        self.worse.clear()
        self.undirected_graph.clear()

        graph_filenames: set[str] = set()
        for comp in comparisons:
            a = comp["filename_a"]
            b = comp["filename_b"]
            winner = comp["winner"]

            if winner == a:
                loser = b
            else:
                loser = a

            if winner not in self.better[loser]:
                self.better[loser].append(winner)
            if loser not in self.worse[winner]:
                self.worse[winner].append(loser)

            self.undirected_graph[a].add(b)
            self.undirected_graph[b].add(a)
            graph_filenames.add(a)
            graph_filenames.add(b)

        if all_filenames is not None:
            self._all_filenames = all_filenames
        else:
            self._all_filenames = graph_filenames

        self._identify_top_bottom()
        self._calculate_chain_lengths()
        self._build_components()

    # ------------------------------------------------------------------
    # Internal build helpers
    # ------------------------------------------------------------------

    def _identify_top_bottom(self) -> None:
        self.top_nodes = [
            img for img in self._all_filenames if not self.better[img]
        ]
        self.bottom_nodes = [
            img for img in self._all_filenames if not self.worse[img]
        ]

    def _calculate_chain_lengths(self) -> None:
        """Calculate chain_length for each image as longest chain containing the node.

        chain_length = longest_ending_at[node] + longest_starting_from[node].
        Nodes with no comparisons get chain_length 0.
        """
        longest_ending_at: dict[str, int] = {}

        def dfs_ending_at(node: str, memo: dict[str, int], visiting: set[str]) -> int:
            if node in memo:
                return memo[node]
            if node in visiting:
                return 0
            visiting.add(node)
            predecessors = self.better.get(node, [])
            if not predecessors:
                memo[node] = 0
            else:
                max_path = 0
                for pred in predecessors:
                    path_len = dfs_ending_at(pred, memo, visiting)
                    max_path = max(max_path, path_len)
                memo[node] = max_path + 1
            visiting.remove(node)
            return memo[node]

        memo_starting: dict[str, int] = {}

        def dfs_starting_from(node: str, memo: dict[str, int], visiting: set[str]) -> int:
            if node in memo:
                return memo[node]
            if node in visiting:
                return 0
            visiting.add(node)
            successors = self.worse.get(node, [])
            if not successors:
                memo[node] = 0
            else:
                max_path = 0
                for succ in successors:
                    path_len = dfs_starting_from(succ, memo, visiting)
                    max_path = max(max_path, path_len)
                memo[node] = max_path + 1
            visiting.remove(node)
            return memo[node]

        memo_ending: dict[str, int] = {}
        for img in self._all_filenames:
            if img not in memo_ending:
                dfs_ending_at(img, memo_ending, set())

        memo_starting: dict[str, int] = {}
        for img in self._all_filenames:
            if img not in memo_starting:
                dfs_starting_from(img, memo_starting, set())

        self.chain_length = {}
        for img in self._all_filenames:
            has_comparisons = img in self.worse or img in self.better
            if not has_comparisons:
                self.chain_length[img] = 0
            else:
                ending = memo_ending.get(img, 0)
                starting = memo_starting.get(img, 0)
                self.chain_length[img] = ending + starting

        self._build_nodes_by_chain_length()

    def _build_nodes_by_chain_length(self) -> None:
        self.nodes_by_chain_length = defaultdict(list)
        for img, cl in self.chain_length.items():
            self.nodes_by_chain_length[cl].append(img)

    def _build_components(self) -> None:
        """Build connected components using BFS on undirected graph."""
        visited: set[str] = set()
        self.component_by_filename = {}
        self.chain_members_by_component = {}
        component_id = 0

        for filename in self._all_filenames:
            if filename in visited:
                continue

            queue = deque([filename])
            visited.add(filename)
            members: list[str] = []

            while queue:
                current = queue.popleft()
                members.append(current)

                for neighbor in self.undirected_graph.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            self.chain_members_by_component[component_id] = members
            for member in members:
                self.component_by_filename[member] = component_id
            component_id += 1

    # ------------------------------------------------------------------
    # Chain enumeration
    # ------------------------------------------------------------------

    def enumerate_all_chains(
        self,
        max_chains: int = 20000000,
        max_depth: int = 50000,
    ) -> list[list[str]]:
        """Enumerate ALL simple directed paths (chains) of length >= 2.

        This is an exhaustive DFS from every node, collecting every path >= 2 edges.
        Used by score redistribution (redistribute_chains.ipynb).
        """
        chains: list[list[str]] = []
        stack: list[tuple[str, list[str]]] = []
        for start in list(self._all_filenames):
            stack.append((start, [start]))
            while stack and len(chains) < max_chains:
                node, path = stack.pop()
                for nb in self.worse.get(node, []):
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
        """Enumerate full top-to-leaf directed chains (only root-to-leaf, no sub-paths)."""
        chains: list[list[str]] = []
        for top in self.top_nodes:
            stack = [(top, [top])]
            while stack:
                node, path = stack.pop()
                worse_nodes = self.worse.get(node, [])
                if not worse_nodes:
                    chains.append(path)
                else:
                    for w in worse_nodes:
                        if w not in path:
                            stack.append((w, path + [w]))
        return chains

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

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
            for w in self.worse.get(current, []):
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
        for w in self.worse.get(u, []):
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
        for chain_len, nodes in self.nodes_by_chain_length.items():
            if len(nodes) > 1:
                top_at_len = [n for n in nodes if not self.better[n]]
                bottom_at_len = [n for n in nodes if not self.worse[n]]

                if len(top_at_len) > 1:
                    for i in range(0, len(top_at_len) - 1, 2):
                        if i + 1 < len(top_at_len):
                            pairs.append((top_at_len[i], top_at_len[i + 1]))

                if len(bottom_at_len) > 1:
                    for i in range(0, len(bottom_at_len) - 1, 2):
                        if i + 1 < len(bottom_at_len):
                            pairs.append((bottom_at_len[i], bottom_at_len[i + 1]))
        return pairs

    def get_max_chain_length(self) -> int:
        return max(self.nodes_by_chain_length.keys()) if self.nodes_by_chain_length else 0

    def get_images_by_chain_length(self, chain_len: int) -> list[str]:
        return self.nodes_by_chain_length.get(chain_len, [])

    def get_top_nodes(self) -> list[str]:
        return self.top_nodes

    def get_bottom_nodes(self) -> list[str]:
        return self.bottom_nodes

    def get_all_chain_lengths(self) -> dict[int, list[str]]:
        return dict(self.nodes_by_chain_length)

    def get_image_chain_info(self, filename: str) -> dict[str, Any] | None:
        if filename not in self.chain_length:
            return None
        cl = self.chain_length[filename]
        nodes_at_len = self.nodes_by_chain_length.get(cl, [])
        position = nodes_at_len.index(filename) if filename in nodes_at_len else 0
        return {
            "filename": filename,
            "chain_length": cl,
            "position": position,
            "better": self.better.get(filename, []),
            "worse": self.worse.get(filename, []),
            "component": self.component_by_filename.get(filename),
        }

    def get_chain_stats(self) -> dict[str, Any]:
        chain_lens = self.nodes_by_chain_length.keys()
        return {
            "max_chain_length": max(chain_lens) if chain_lens else 0,
            "min_chain_length": min(chain_lens) if chain_lens else 0,
            "top_nodes_count": len(self.top_nodes),
            "bottom_nodes_count": len(self.bottom_nodes),
            "nodes_by_chain_length": {cl: len(n) for cl, n in self.nodes_by_chain_length.items()},
        }


class CrystalGraph:
    """Manages a crystal-like tournament graph from comparison history.

    Owns database/cache metadata and delegates all chain computation to
    ChainManager (self.chain).  Exposes chain attributes via properties
    for backward compatibility.
    """

    def __init__(self) -> None:
        self.images: dict[str, dict[str, Any]] = {}
        self.comparisons: list[dict[str, Any]] = []
        self.chain = ChainManager()
        self._built_at: datetime | None = None
        self._db_comparison_count: int = 0

    # -- Chain attribute pass-through properties -------------------------

    @property
    def better(self) -> defaultdict[str, list[str]]:
        return self.chain.better

    @property
    def worse(self) -> defaultdict[str, list[str]]:
        return self.chain.worse

    @property
    def chain_length(self) -> dict[str, int]:
        """Longest chain through each node.  Canonical name."""
        return self.chain.chain_length

    @property
    def height(self) -> dict[str, int]:
        """Deprecated alias for chain_length — kept for backward compat."""
        return self.chain.chain_length

    @property
    def nodes_by_chain_length(self) -> dict[int, list[str]]:
        return self.chain.nodes_by_chain_length

    @property
    def nodes_by_height(self) -> dict[int, list[str]]:
        """Deprecated alias for nodes_by_chain_length."""
        return self.chain.nodes_by_chain_length

    @property
    def top_nodes(self) -> list[str]:
        return self.chain.top_nodes

    @property
    def bottom_nodes(self) -> list[str]:
        return self.chain.bottom_nodes

    @property
    def undirected_graph(self) -> defaultdict[str, set[str]]:
        return self.chain.undirected_graph

    @property
    def component_by_filename(self) -> dict[str, int]:
        return self.chain.component_by_filename

    @property
    def chain_members_by_component(self) -> dict[int, list[str]]:
        return self.chain.chain_members_by_component

    # -- Build / refresh -------------------------------------------------

    def build_from_database(
        self,
        images: list[dict[str, Any]] | None = None,
        comparisons: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build the crystal graph from database comparison history."""
        from external_modules.step01ranking_new.database.images_table import get_all_images
        from external_modules.step01ranking_new.database.comparisons_table import (
            get_all_comparisons,
        )

        if images is None:
            images = get_all_images()
        if comparisons is None:
            comparisons = get_all_comparisons()

        self.images = {img["filename"]: img for img in images}
        self.comparisons = comparisons
        self._db_comparison_count = len(comparisons)
        self._built_at = datetime.now(timezone.utc)

        all_filenames: set[str] = set(self.images.keys())
        for comp in comparisons:
            all_filenames.add(comp["filename_a"])
            all_filenames.add(comp["filename_b"])

        self.chain.build(comparisons, all_filenames=all_filenames)

    # -- Delegated methods -----------------------------------------------

    def _enumerate_all_chains(self) -> list[list[str]]:
        return self.chain.enumerate_top_chains()

    def get_image_info(self, filename: str) -> dict[str, Any] | None:
        info = self.chain.get_image_chain_info(filename)
        return info

    def get_images_by_height(self, height: int) -> list[str]:
        """Deprecated — prefer get_images_by_chain_length."""
        return self.chain.get_images_by_chain_length(height)

    def get_images_by_chain_length(self, cl: int) -> list[str]:
        return self.chain.get_images_by_chain_length(cl)

    def get_max_height(self) -> int:
        """Deprecated — prefer get_max_chain_length."""
        return self.chain.get_max_chain_length()

    def get_max_chain_length(self) -> int:
        return self.chain.get_max_chain_length()

    def get_collapsable_pairs(self) -> list[tuple[str, str]]:
        return self.chain.get_collapsable_pairs()

    def get_top_nodes(self) -> list[str]:
        return self.chain.get_top_nodes()

    def get_bottom_nodes(self) -> list[str]:
        return self.chain.get_bottom_nodes()

    def get_all_heights(self) -> dict[int, list[str]]:
        """Deprecated — prefer get_all_chain_lengths."""
        return self.chain.get_all_chain_lengths()

    def get_all_chain_lengths(self) -> dict[int, list[str]]:
        return self.chain.get_all_chain_lengths()

    def are_in_same_path(self, img1: str, img2: str) -> bool:
        return self.chain.are_in_same_path(img1, img2)

    def is_redundant(
        self,
        u: str,
        v: str,
        skip_edges: set[tuple[str, str]] | None = None,
    ) -> bool:
        return self.chain.is_redundant(u, v, skip_edges)

    def _can_reach(
        self,
        start: str,
        end: str,
        skip_edges: set[tuple[str, str]] | None = None,
    ) -> bool:
        return self.chain._can_reach(start, end, skip_edges)

    def get_graph_stats(self) -> dict[str, Any]:
        stats = self.chain.get_chain_stats()
        stats.update({
            "total_images": len(self.images),
            "total_comparisons": len(self.comparisons),
            "built_at": self._built_at.isoformat() if self._built_at else None,
        })
        return stats

    # -- Cache management ------------------------------------------------

    def is_cache_stale(self) -> bool:
        if self._built_at is None:
            return True
        from external_modules.step01ranking_new.database.comparisons_table import (
            get_total_comparisons,
        )
        current_count = get_total_comparisons()
        return current_count != self._db_comparison_count

    def get_cache_info(self) -> dict[str, Any]:
        return {
            "built_at": self._built_at.isoformat() if self._built_at else None,
            "comparison_count_at_build": self._db_comparison_count,
            "is_stale": self.is_cache_stale(),
        }


crystal_graph = CrystalGraph()
crystal_graph.build_from_database()
