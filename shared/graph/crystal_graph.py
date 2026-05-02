from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any


class CrystalGraph:
    """Manages a crystal-like tournament graph from comparison history.

    In this structure:
    - Each image can have multiple 'better' (images that beat it) and 'worse' (images it beat)
    - Height represents the length of the longest top-to-leaf chain containing the node
    - Two images are in the same path if one can reach the other via directed winner->loser edges

    The graph is cached in memory after build_from_database() is called.
    Call build_from_database() again to refresh from the database.
    """

    def __init__(self) -> None:
        self.images: dict[str, dict[str, Any]] = {}
        self.comparisons: list[dict[str, Any]] = []
        self.better: defaultdict[str, list[str]] = defaultdict(list)
        self.worse: defaultdict[str, list[str]] = defaultdict(list)
        self.height: dict[str, int] = {}
        self.nodes_by_height: dict[int, list[str]] = defaultdict(list)
        self.top_nodes: list[str] = []
        self.bottom_nodes: list[str] = []
        self.undirected_graph: defaultdict[str, set[str]] = defaultdict(set)
        self.component_by_filename: dict[str, int] = {}
        self.chain_members_by_component: dict[int, list[str]] = {}
        self._built_at: datetime | None = None
        self._db_comparison_count: int = 0

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

        self.better = defaultdict(list)
        self.worse = defaultdict(list)
        self.height = {}
        self.undirected_graph = defaultdict(set)

        for comp in comparisons:
            filename_a = comp["filename_a"]
            filename_b = comp["filename_b"]
            winner = comp["winner"]

            if winner == filename_a:
                loser = filename_b
            else:
                loser = filename_a

            if winner not in self.better[loser]:
                self.better[loser].append(winner)
            if loser not in self.worse[winner]:
                self.worse[winner].append(loser)

            # Build undirected graph for component detection
            self.undirected_graph[filename_a].add(filename_b)
            self.undirected_graph[filename_b].add(filename_a)

        self._identify_top_bottom()
        self._calculate_heights()
        self._build_components()

    def _enumerate_all_chains(self) -> list[list[str]]:
        """Enumerate all directed chains from top nodes to leaves."""
        chains = []
        for top in self.top_nodes:
            stack = [(top, [top])]
            while stack:
                node, path = stack.pop()
                worse_nodes = self.worse.get(node, [])
                if not worse_nodes:
                    chains.append(path)
                else:
                    for worse in worse_nodes:
                        if worse not in path:
                            stack.append((worse, path + [worse]))
        return chains

    def _calculate_heights(self) -> None:
        """Calculate height for each image as max length of chains containing it."""
        self.height = {img: 0 for img in self.images}
        chains = self._enumerate_all_chains()

        for chain in chains:
            chain_len = len(chain)
            for node in chain:
                if self.height[node] < chain_len:
                    self.height[node] = chain_len

        self._build_nodes_by_height()

    def _build_nodes_by_height(self) -> None:
        """Group nodes by their height."""
        self.nodes_by_height = defaultdict(list)
        for img, h in self.height.items():
            self.nodes_by_height[h].append(img)

    def _build_components(self) -> None:
        """Build connected components using BFS on undirected graph."""
        visited: set[str] = set()
        self.component_by_filename = {}
        self.chain_members_by_component = {}
        component_id = 0

        # Include all images, not just those in comparisons
        all_filenames = set(self.images.keys())
        for filename in self.undirected_graph:
            all_filenames.add(filename)

        for filename in all_filenames:
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

    def _identify_top_bottom(self) -> None:
        """Identify top nodes (no better) and bottom nodes (no worse)."""
        self.top_nodes = [img for img in self.images if not self.better[img]]
        self.bottom_nodes = [img for img in self.images if not self.worse[img]]

    def get_image_info(self, filename: str) -> dict[str, Any] | None:
        """Get height and position information for a given image."""
        if filename not in self.height:
            return None

        height = self.height[filename]
        nodes_at_height = self.nodes_by_height.get(height, [])
        position = nodes_at_height.index(filename) if filename in nodes_at_height else 0

        return {
            "filename": filename,
            "height": height,
            "position": position,
            "better": self.better.get(filename, []),
            "worse": self.worse.get(filename, []),
            "component": self.component_by_filename.get(filename),
        }

    def get_images_by_height(self, height: int) -> list[str]:
        """Return a list of images with a given height."""
        return self.nodes_by_height.get(height, [])

    def get_max_height(self) -> int:
        """Get the maximum height in the crystal graph."""
        return max(self.nodes_by_height.keys()) if self.nodes_by_height else 0

    def get_collapsable_pairs(self) -> list[tuple[str, str]]:
        """Find pairs of nodes that need to be compared to collapse the graph."""
        pairs = []
        for height, nodes in self.nodes_by_height.items():
            if len(nodes) > 1:
                top_at_height = [n for n in nodes if not self.better[n]]
                bottom_at_height = [n for n in nodes if not self.worse[n]]

                if len(top_at_height) > 1:
                    for i in range(0, len(top_at_height) - 1, 2):
                        if i + 1 < len(top_at_height):
                            pairs.append((top_at_height[i], top_at_height[i + 1]))

                if len(bottom_at_height) > 1:
                    for i in range(0, len(bottom_at_height) - 1, 2):
                        if i + 1 < len(bottom_at_height):
                            pairs.append((bottom_at_height[i], bottom_at_height[i + 1]))
        return pairs

    def get_top_nodes(self) -> list[str]:
        """Get all top nodes (nodes with no better connection)."""
        return self.top_nodes

    def get_bottom_nodes(self) -> list[str]:
        """Get all bottom nodes (nodes with no worse connections)."""
        return self.bottom_nodes

    def get_all_heights(self) -> dict[int, list[str]]:
        """Get all height levels with their nodes."""
        return dict(self.nodes_by_height)

    def get_graph_stats(self) -> dict[str, Any]:
        """Get statistics about the current graph."""
        heights = self.nodes_by_height.keys()
        return {
            "total_images": len(self.images),
            "total_comparisons": len(self.comparisons),
            "max_height": max(heights) if heights else 0,
            "min_height": min(heights) if heights else 0,
            "top_nodes_count": len(self.top_nodes),
            "bottom_nodes_count": len(self.bottom_nodes),
            "nodes_by_height": {h: len(n) for h, n in self.nodes_by_height.items()},
            "built_at": self._built_at.isoformat() if self._built_at else None,
        }

    def _can_reach(self, start: str, end: str) -> bool:
        """Check if end is reachable from start via worse edges (directed)."""
        if start not in self.images or end not in self.images:
            return False
        visited = set()
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current == end:
                return True
            visited.add(current)
            for worse in self.worse.get(current, []):
                if worse not in visited:
                    queue.append(worse)
        return False

    def are_in_same_path(self, img1: str, img2: str) -> bool:
        """Check if two images are in same directed path (one can reach the other)."""
        if img1 not in self.images or img2 not in self.images:
            return False
        if img1 == img2:
            return True
        if self._can_reach(img1, img2):
            return True
        if self._can_reach(img2, img1):
            return True
        return False

    def is_cache_stale(self) -> bool:
        """Check if the cached graph is stale (database has new comparisons)."""
        if self._built_at is None:
            return True
        from external_modules.step01ranking_new.database.comparisons_table import (
            get_total_comparisons,
        )
        current_count = get_total_comparisons()
        return current_count != self._db_comparison_count

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about the cached graph."""
        return {
            "built_at": self._built_at.isoformat() if self._built_at else None,
            "comparison_count_at_build": self._db_comparison_count,
            "is_stale": self.is_cache_stale(),
        }


crystal_graph = CrystalGraph()
crystal_graph.build_from_database()
