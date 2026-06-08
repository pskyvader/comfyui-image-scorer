import sys
from pathlib import Path
from collections import defaultdict, deque

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from ...graph.crystal_graph import (
    crystal_graph,
    CrystalGraph,
    NodeProxy,
    ChainProxy,
    ComponentProxy,
    ChainManager,
)
from ....external_modules.database_structure.images_table import get_all_images

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_graph(edges: list[tuple[str, str]], images: list[str] | None = None):
    """Rebuild the singleton crystal_graph from a list of (winner, loser) edges.

    If *images* is not supplied every filename mentioned in *edges* is used.
    """
    if images is None:
        images = sorted({n for e in edges for n in e})
    crystal_graph.rebuild_from_database(
        images=[{"filename": n} for n in images],
        comparisons=[{"filename_a": a, "filename_b": b, "winner": a} for a, b in edges],
    )
    return crystal_graph


def has_cycle(edges: list[tuple[str, str]]) -> bool:
    """Return True if the directed graph defined by *edges* contains a cycle."""
    adj: dict[str, list[str]] = defaultdict(list)
    nodes: set[str] = set()
    for a, b in edges:
        adj[a].append(b)
        nodes.add(a)
        nodes.add(b)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in nodes}

    def dfs(u: str) -> bool:
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    return any(dfs(n) for n in nodes if color[n] == WHITE)


def topological_sort_nodes(edges: list[tuple[str, str]]) -> list[str] | None:
    """Topological sort via Kahn's algorithm. Returns None if a cycle exists."""
    adj: dict[str, list[str]] = defaultdict(list)
    indeg: dict[str, int] = defaultdict(int)
    nodes: set[str] = set()
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
        nodes.add(a)
        nodes.add(b)

    queue = deque([n for n in nodes if indeg[n] == 0])
    order: list[str] = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    return order if len(order) == len(nodes) else None


def test_orphan_nodes():
    """check if theres any orphan node with comparisons, which should not happen"""
    all_images = get_all_images()
    images_nodes = {row["filename"]: row for row in all_images}
    all_nodes: list[NodeProxy] = crystal_graph.get_all_nodes()
    with tqdm(total=len(all_nodes)):
        for node in all_nodes:
            if int(images_nodes[node.filename]["comparison_count"]) > 0:
                main_chain: ChainProxy = node.get_chain()[0]
                assert len(main_chain.get_nodes()) > 0


# ---------------------------------------------------------------------------
# Test 1: Every main chain has at least one node that claims it as main
# ---------------------------------------------------------------------------


class TestMainChainHasAtLeastOneClaimant:
    """Every raw chain from the min-chain-cover algorithm must be somebody's
    main chain (i.e. every chain must have at least one main node)."""

    def _assert_all_raw_chains_claimed(self, g: CrystalGraph):
        raw = g._chain.compute_min_chains()
        raw_ids: set[int] = set(range(len(raw)))

        chains_map = g.get_chains_map()
        claimed: set[int] = set()
        for length_data in chains_map.values():
            for chain_id, (chain, nodes) in length_data.items():
                if any(is_main for _, is_main in nodes):
                    claimed.add(chain_id)

        assert raw_ids == claimed, f"Unclaimed chains: {raw_ids - claimed}"

    def test_single_chain(self):
        self._assert_all_raw_chains_claimed(build_graph([("a", "b"), ("b", "c")]))

    def test_diamond_graph(self):
        self._assert_all_raw_chains_claimed(
            build_graph([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
        )

    def test_two_separate_components(self):
        self._assert_all_raw_chains_claimed(build_graph([("a", "b"), ("c", "d")]))

    def test_wide_diamond(self):
        self._assert_all_raw_chains_claimed(
            build_graph(
                [
                    ("a", "b"),
                    ("a", "c"),
                    ("a", "d"),
                    ("b", "e"),
                    ("c", "e"),
                    ("d", "e"),
                ]
            )
        )

    def test_complex_merge(self):
        self._assert_all_raw_chains_claimed(
            build_graph(
                [
                    ("a", "b"),
                    ("b", "c"),
                    ("c", "d"),
                    ("e", "f"),
                    ("f", "c"),
                ]
            )
        )


# ---------------------------------------------------------------------------
# Test 2: All nodes with comparison_length > 0 must have a main chain
# ---------------------------------------------------------------------------


class TestNodesWithComparisonsHaveMainChain:
    """Any node that participates in at least one comparison must belong to
    at least one chain (i.e. node.get_chain(only_main=True) succeeds)."""

    def test_linear_chain(self):
        g = build_graph([("a", "b"), ("b", "c"), ("c", "d")])
        for name in ("a", "b", "c", "d"):
            node = g.get_node(name)
            chains = node.get_chain(only_main=True)
            assert len(chains) >= 1, f"Node {name} has no main chain"

    def test_diamond(self):
        g = build_graph([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
        for name in ("a", "b", "c", "d"):
            node = g.get_node(name)
            chains = node.get_chain(only_main=True)
            assert len(chains) >= 1, f"Node {name} has no main chain"

    def test_two_separate_components(self):
        g = build_graph([("a", "b"), ("c", "d")])
        for name in ("a", "b", "c", "d"):
            node = g.get_node(name)
            chains = node.get_chain(only_main=True)
            assert len(chains) >= 1, f"Node {name} has no main chain"

    def test_isolated_nodes_excluded(self):
        g = build_graph([], images=["x", "y"])
        for name in ("x", "y"):
            node = g.get_node(name)
            length = g._chain._chain_length.get(name, 0)
            assert length == 0

    def test_mixed_isolated_and_compared(self):
        g = build_graph([("a", "b")], images=["a", "b", "c"])
        for name in ("a", "b"):
            node = g.get_node(name)
            chains = node.get_chain(only_main=True)
            assert len(chains) >= 1, f"Node {name} has no main chain"
        c_node = g.get_node("c")
        assert g._chain._chain_length.get("c", 0) == 0

    def test_star_topology(self):
        g = build_graph(
            [
                ("a", "b"),
                ("a", "c"),
                ("a", "d"),
                ("a", "e"),
            ]
        )
        for name in ("a", "b", "c", "d", "e"):
            node = g.get_node(name)
            chains = node.get_chain(only_main=True)
            assert len(chains) >= 1, f"Node {name} has no main chain"

    def test_very_long_chain(self):
        nodes = [str(i) for i in range(20)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(19)]
        g = build_graph(edges)
        for name in nodes:
            node = g.get_node(name)
            chains = node.get_chain(only_main=True)
            assert len(chains) >= 1, f"Node {name} has no main chain"

    def test_merge_point_in_multiple_chains(self):
        g = build_graph(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("e", "f"),
                ("f", "c"),
            ]
        )
        node_c = g.get_node("c")
        main = node_c.get_chain(only_main=True)
        assert len(main) == 1
        all_chains = node_c.get_chain(only_main=False)
        assert len(all_chains) >= 1, "c should appear in at least one chain"

    def test_wide_graph_all_nodes_covered(self):
        g = build_graph(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("e", "f"),
                ("f", "g"),
                ("g", "d"),
            ]
        )
        for name in ("a", "b", "c", "d", "e", "f", "g"):
            node = g.get_node(name)
            chains = node.get_chain(only_main=True)
            assert len(chains) >= 1, f"Node {name} has no main chain"


# ---------------------------------------------------------------------------
# Test 3: Single-graph constraint
# ---------------------------------------------------------------------------


class TestSingleGraphTopBottom:
    """For single-component single-source single-sink graphs, verify the
    fundamental invariant: every node's main chain is the longest chain
    containing that node."""

    def _assert_single_graph_property(self, g: CrystalGraph):
        top_nodes = g._chain._top_nodes
        bottom_nodes = g._chain._bottom_nodes
        if len(top_nodes) != 1 or len(bottom_nodes) != 1:
            return

        for node in g.get_all_nodes():
            main = node.get_chain(only_main=True)
            all_chains = node.get_chain(only_main=False)
            if all_chains:
                main_len = main[0].length
                max_len = max(c.length for c in all_chains)
                assert main_len == max_len, (
                    f"Node {node.filename}: main chain length {main_len} "
                    f"!= max chain length {max_len}"
                )

    def test_linear(self):
        g = build_graph([("a", "b"), ("b", "c"), ("c", "d")])
        self._assert_single_graph_property(g)

    def test_diamond(self):
        g = build_graph([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
        self._assert_single_graph_property(g)

    def test_wide_diamond(self):
        g = build_graph(
            [
                ("a", "b"),
                ("a", "c"),
                ("a", "d"),
                ("b", "e"),
                ("c", "e"),
                ("d", "e"),
            ]
        )
        self._assert_single_graph_property(g)

    def test_complex_single_source_single_sink(self):
        g = build_graph(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("a", "e"),
                ("e", "c"),
                ("a", "f"),
                ("f", "d"),
            ]
        )
        self._assert_single_graph_property(g)

    def test_two_components_not_applicable(self):
        g = build_graph([("a", "b"), ("c", "d")])
        assert len(g._chain._top_nodes) == 2
        assert len(g._chain._bottom_nodes) == 2

    def test_single_comparison(self):
        g = build_graph([("a", "b")])
        self._assert_single_graph_property(g)

    def test_three_node_chain(self):
        g = build_graph([("x", "y"), ("y", "z")])
        self._assert_single_graph_property(g)


# ---------------------------------------------------------------------------
# Test 4: Cycle (loop) detection
# ---------------------------------------------------------------------------


class TestCycleDetection:
    """Detect directed cycles in the comparison graph."""

    def test_no_cycle_linear(self):
        edges = [("a", "b"), ("b", "c"), ("c", "d")]
        assert not has_cycle(edges)

    def test_no_cycle_diamond(self):
        edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
        assert not has_cycle(edges)

    def test_cycle_two_nodes(self):
        edges = [("a", "b"), ("b", "a")]
        assert has_cycle(edges)

    def test_cycle_three_nodes(self):
        edges = [("a", "b"), ("b", "c"), ("c", "a")]
        assert has_cycle(edges)

    def test_cycle_in_larger_graph(self):
        edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "b")]
        assert has_cycle(edges)

    def test_self_loop(self):
        edges = [("a", "a")]
        assert has_cycle(edges)

    def test_no_cycle_disconnected(self):
        edges = [("a", "b"), ("c", "d")]
        assert not has_cycle(edges)

    def test_topological_sort_confirms_cycle(self):
        edges = [("a", "b"), ("b", "c"), ("c", "a")]
        assert topological_sort_nodes(edges) is None

    def test_topological_sort_confirms_no_cycle(self):
        edges = [("a", "b"), ("b", "c"), ("c", "d")]
        order = topological_sort_nodes(edges)
        assert order is not None
        assert len(order) == 4

    def test_build_graph_with_cycle_does_not_crash(self):
        g = build_graph(
            [("a", "b"), ("b", "c"), ("c", "a")],
            images=["a", "b", "c"],
        )
        chains = g.get_all_chains()
        assert isinstance(chains, list)


# ---------------------------------------------------------------------------
# Test 5: Main chains must not contain loops
# ---------------------------------------------------------------------------


class TestMainChainsAreAcyclic:
    """Even if the comparison graph contains cycles, each individual chain
    must not visit the same node twice."""

    def test_all_chains_acyclic_linear(self):
        g = build_graph([("a", "b"), ("b", "c"), ("c", "d")])
        for chain in g.get_all_chains():
            nodes = [n.filename for n in chain.nodes]
            assert len(nodes) == len(
                set(nodes)
            ), f"Chain {chain.id} has duplicate nodes: {nodes}"

    def test_all_chains_acyclic_diamond(self):
        g = build_graph([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
        for chain in g.get_all_chains():
            nodes = [n.filename for n in chain.nodes]
            assert len(nodes) == len(
                set(nodes)
            ), f"Chain {chain.id} has duplicate nodes: {nodes}"

    def test_all_chains_acyclic_merge(self):
        g = build_graph(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("e", "f"),
                ("f", "c"),
            ]
        )
        for chain in g.get_all_chains():
            nodes = [n.filename for n in chain.nodes]
            assert len(nodes) == len(
                set(nodes)
            ), f"Chain {chain.id} has duplicate nodes: {nodes}"

    def test_all_chains_acyclic_wide(self):
        g = build_graph(
            [
                ("a", "b"),
                ("a", "c"),
                ("a", "d"),
                ("b", "e"),
                ("c", "e"),
                ("d", "e"),
            ]
        )
        for chain in g.get_all_chains():
            nodes = [n.filename for n in chain.nodes]
            assert len(nodes) == len(
                set(nodes)
            ), f"Chain {chain.id} has duplicate nodes: {nodes}"

    def test_enumerate_top_chains_acyclic(self):
        g = build_graph(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("a", "e"),
                ("e", "d"),
            ]
        )
        for chain in g._chain.enumerate_top_chains():
            assert len(chain) == len(
                set(chain)
            ), f"Top chain has duplicate nodes: {chain}"

    def test_min_chain_cover_covers_all_nodes(self):
        g = build_graph(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("e", "f"),
                ("f", "c"),
            ]
        )
        min_chains = g._chain.compute_min_chains()
        covered = set()
        for chain in min_chains:
            covered.update(chain)
        for name in ("a", "b", "c", "d", "e", "f"):
            assert name in covered, f"Node {name} not covered by any min chain"

    def test_chain_respects_dag_order(self):
        g = build_graph(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("a", "e"),
                ("e", "d"),
            ]
        )
        for chain in g.get_all_chains():
            nodes = [n.filename for n in chain.nodes]
            for i in range(len(nodes) - 1):
                x, y = nodes[i], nodes[i + 1]
                assert y in g._chain._worse_than.get(
                    x, []
                ), f"Chain {chain.id}: {x} -> {y} is not a valid edge"


# ---------------------------------------------------------------------------
# Test 6: Additional useful tests
# ---------------------------------------------------------------------------


class TestAdditionalProperties:
    """Miscellaneous sanity checks on the crystal graph."""

    def test_empty_graph(self):
        g = build_graph([], images=[])
        assert g.get_all_chains() == []
        assert len(g.get_all_nodes()) == 0

    def test_single_node(self):
        g = build_graph([], images=["a"])
        assert len(g.get_all_nodes()) == 1
        chains = g.get_all_chains()

    def test_two_nodes_one_comparison(self):
        g = build_graph([("a", "b")])
        chains = g.get_all_chains()
        assert len(chains) >= 1
        all_nodes_in_chains = set()
        for c in chains:
            all_nodes_in_chains.update(n.filename for n in c.nodes)
        assert "a" in all_nodes_in_chains
        assert "b" in all_nodes_in_chains

    def test_graph_stats(self):
        g = build_graph([("a", "b"), ("b", "c")])
        stats = g.get_graph_stats()
        assert stats["total_images"] == 3
        assert stats["total_comparisons"] == 2
        assert stats["total_components"] == 1
        assert stats["total_chains"] >= 1
        assert stats["longest_chain_depth"] >= 2

    def test_components_count(self):
        g = build_graph([("a", "b"), ("c", "d"), ("e", "f")])
        assert len(g.get_all_components()) == 3

    def test_component_sizes(self):
        g = build_graph([("a", "b"), ("b", "c"), ("d", "e")])
        comps = g.get_all_components()
        sizes = sorted(c.size for c in comps)
        assert sizes == [2, 3]

    def test_top_nodes_count(self):
        g = build_graph([("a", "b"), ("a", "c"), ("a", "d")])
        assert len(g._chain._top_nodes) == 1
        assert g._chain._top_nodes[0] == "a"

    def test_bottom_nodes_count(self):
        g = build_graph([("b", "a"), ("c", "a"), ("d", "a")])
        assert len(g._chain._bottom_nodes) == 1
        assert g._chain._bottom_nodes[0] == "a"

    def test_get_node_proxy(self):
        g = build_graph([("a", "b")])
        node = g.get_node("a")
        assert isinstance(node, NodeProxy)
        assert node.filename == "a"
        assert node.is_top()
        assert not node.is_bottom()

        node_b = g.get_node("b")
        assert not node_b.is_top()
        assert node_b.is_bottom()

    def test_get_nonexistent_node(self):
        g = build_graph([("a", "b")])
        assert g.get_node("z") is None
        assert g.get_node(None) is None

    def test_get_chain_by_node(self):
        g = build_graph([("a", "b"), ("b", "c")])
        chain = g.get_chain(node_id="a")
        assert isinstance(chain, ChainProxy)
        assert chain.length == 3

    def test_get_chain_by_id(self):
        g = build_graph([("a", "b")])
        chain = g.get_chain(chain_id=0)
        assert isinstance(chain, ChainProxy)

    def test_get_chain_invalid_id(self):
        g = build_graph([("a", "b")])
        assert g.get_chain(chain_id=999) is None

    def test_get_chain_no_args_raises(self):
        g = build_graph([("a", "b")])
        import pytest

        with pytest.raises(ValueError):
            g.get_chain()

    def test_chain_proxy_properties(self):
        g = build_graph([("a", "b"), ("b", "c"), ("c", "d")])
        chain = g.get_chain(node_id="a")
        assert chain.first.filename == "a"
        assert chain.last.filename == "d"
        assert chain.length == 4
        assert chain.node_position("a") == 0
        assert chain.node_position("d") == 3

    def test_node_get_links(self):
        g = build_graph([("a", "b"), ("a", "c")])
        node_a = g.get_node("a")
        worse = node_a.get_links(worse_than=True)
        assert len(worse) == 2
        better = node_a.get_links(better_than=True)
        assert len(better) == 0

        node_b = g.get_node("b")
        b_better = node_b.get_links(better_than=True)
        assert len(b_better) == 1

    def test_are_in_same_path(self):
        g = build_graph([("a", "b"), ("b", "c")])
        assert g.are_in_same_path("a", "c") is True
        assert g.are_in_same_path("a", "a") is True

    def test_are_in_same_path_false(self):
        g = build_graph([("a", "b"), ("c", "b")])
        assert g.are_in_same_path("a", "c") is False

    def test_get_all_links(self):
        g = build_graph([("a", "b"), ("a", "c")])
        links = g.get_all_links()
        assert len(links) == 2

    def test_is_redundant(self):
        g = build_graph([("a", "b"), ("b", "c")])
        assert g.is_redundant("a", "b") is False
        assert g.is_redundant("b", "c") is False

    def test_get_collapsable_pairs(self):
        g = build_graph([("a", "b"), ("b", "c")])
        pairs = g.get_collapsable_pairs()
        assert isinstance(pairs, list)

    def test_apply_comparison_incremental(self):
        g = build_graph([], images=["a"])
        g.apply_comparison("a", "b")
        assert g.get_node("b") is not None
        assert g.get_node("b").is_bottom()
        assert len(g.get_all_nodes()) == 2

    def test_apply_comparison_chain_growth(self):
        g = build_graph([], images=["a"])
        for name in ["b", "c", "d", "e"]:
            g.apply_comparison("a" if name == "b" else chr(ord(name) - 1), name)
        for name in ["a", "b", "c", "d", "e"]:
            length = g._chain._chain_length.get(name, 0)
            assert length == 4, f"Node {name} has chain_length {length}, expected 4"

    def test_apply_comparison_merge_components(self):
        g = build_graph([("a", "b"), ("c", "d")])
        assert len(g.get_all_components()) == 2
        g.apply_comparison("b", "c")
        assert len(g.get_all_components()) == 1

    def test_cache_info(self):
        g = build_graph([("a", "b")])
        info = g.get_cache_info()
        assert info["built_at"] is not None
        assert info["comparison_count_at_build"] == 1
        assert isinstance(info["is_stale"], bool)

    def test_get_duplicate_comparison_ids(self):
        g = build_graph([("a", "b")])
        dupes = g.get_duplicate_comparison_ids()
        assert isinstance(dupes, list)
        assert len(dupes) == 0

    def test_get_redundant_edges(self):
        g = build_graph([("a", "b"), ("b", "c")])
        redundant = g.get_redundant_edges()
        assert isinstance(redundant, set)

    def test_get_component_by_node(self):
        g = build_graph([("a", "b")])
        comp = g.get_component(node_id="a")
        assert isinstance(comp, ComponentProxy)
        assert comp.size == 2

    def test_get_component_by_id(self):
        g = build_graph([("a", "b")])
        comp = g.get_component(node_id="a")
        comp2 = g.get_component(component_id=comp.id)
        assert comp2.id == comp.id

    def test_get_component_nonexistent(self):
        g = build_graph([("a", "b")])
        assert g.get_component(node_id="z") is None
        assert g.get_component(component_id=999) is None

    def test_get_all_sub_chains(self):
        g = build_graph([("a", "b"), ("b", "c")])
        subs = g.get_all_sub_chains(max_chains=100, max_depth=100)
        assert len(subs) >= 3

    def test_chain_cover_minimality(self):
        g = build_graph([("a", "b"), ("b", "c"), ("c", "d")])
        min_chains = g._chain.compute_min_chains()
        assert len(min_chains) <= 4

    def test_every_chain_node_has_valid_edge(self):
        g = build_graph(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("a", "e"),
                ("e", "d"),
            ]
        )
        for chain in g.get_all_chains():
            nodes = [n.filename for n in chain.nodes]
            for i in range(len(nodes) - 1):
                x, y = nodes[i], nodes[i + 1]
                assert y in g._chain._worse_than.get(
                    x, []
                ), f"Chain {chain.id}: no edge {x} -> {y}"

    def test_main_chain_longest_for_node(self):
        g = build_graph(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("a", "e"),
                ("e", "d"),
            ]
        )
        for name in ("a", "b", "c", "d", "e"):
            node = g.get_node(name)
            main = node.get_chain(only_main=True)
            all_chains = node.get_chain(only_main=False)
            if all_chains:
                main_len = main[0].length
                max_len = max(c.length for c in all_chains)
                assert main_len == max_len, (
                    f"Node {name}: main chain length {main_len} "
                    f"!= max chain length {max_len}"
                )

    def test_single_node_no_comparisons(self):
        g = build_graph([], images=["lonely"])
        node = g.get_node("lonely")
        assert g._chain._chain_length.get("lonely", 0) == 0

    def test_many_parallel_chains(self):
        g = build_graph(
            [
                ("a1", "a2"),
                ("a2", "a3"),
                ("b1", "b2"),
                ("b2", "b3"),
                ("c1", "c2"),
                ("c2", "c3"),
                ("d1", "d2"),
                ("d2", "d3"),
            ]
        )
        assert len(g.get_all_components()) == 4
        chains_map = g.get_chains_map()
        all_chains = g.get_all_chains()
        all_ids = {c.id for c in all_chains}
        claimed: set[int] = set()
        for length_data in chains_map.values():
            for chain_id, (chain, nodes) in length_data.items():
                if any(is_main for _, is_main in nodes):
                    claimed.add(chain_id)
        assert all_ids == claimed, f"Unclaimed chains: {all_ids - claimed}"

    def test_complex_dag_all_nodes_in_main_chain(self):
        g = build_graph(
            [
                ("a", "b"),
                ("a", "c"),
                ("a", "d"),
                ("b", "e"),
                ("c", "e"),
                ("c", "f"),
                ("d", "f"),
                ("e", "g"),
                ("f", "g"),
            ]
        )
        for name in ("a", "b", "c", "d", "e", "f", "g"):
            node = g.get_node(name)
            chains = node.get_chain(only_main=True)
            assert len(chains) >= 1, f"Node {name} has no main chain"

    def test_single_source_single_sink_node_main_chain_is_longest(self):
        g = build_graph(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("a", "e"),
                ("e", "d"),
            ]
        )
        for name in ("a", "b", "c", "d", "e"):
            node = g.get_node(name)
            main = node.get_chain(only_main=True)
            all_chains = node.get_chain(only_main=False)
            if all_chains:
                main_len = main[0].length
                max_len = max(c.length for c in all_chains)
                assert main_len == max_len, (
                    f"Node {name}: main chain length {main_len} "
                    f"!= max chain length {max_len}"
                )

    def test_disconnected_graph_all_nodes_have_chains(self):
        g = build_graph(
            [
                ("a", "b"),
                ("c", "d"),
                ("e", "f"),
            ]
        )
        for name in ("a", "b", "c", "d", "e", "f"):
            node = g.get_node(name)
            chains = node.get_chain(only_main=True)
            assert len(chains) >= 1, f"Node {name} has no main chain"


# ---------------------------------------------------------------------------
# Test 7: A node's main chain is always the longest chain containing it
# ---------------------------------------------------------------------------


class TestMainChainIsLongest:
    """A node's main chain (get_chain(only_main=True)) must be the longest
    chain that contains that node.  This is the fundamental invariant of the
    chain-cover algorithm."""

    def _check_all_nodes(self, g: CrystalGraph, names: list[str]):
        for name in names:
            node = g.get_node(name)
            main = node.get_chain(only_main=True)
            all_chains = node.get_chain(only_main=False)
            if not all_chains:
                continue
            main_len = main[0].length
            max_len = max(c.length for c in all_chains)
            assert main_len == max_len, (
                f"Node {name}: main chain length {main_len} != "
                f"max chain length {max_len}.  "
                f"Main chain id={main[0].id}, all ids="
                f"{[c.id for c in all_chains]}"
            )

    def test_linear(self):
        g = build_graph([("a", "b"), ("b", "c"), ("c", "d")])
        self._check_all_nodes(g, ["a", "b", "c", "d"])

    def test_diamond(self):
        g = build_graph([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
        self._check_all_nodes(g, ["a", "b", "c", "d"])

    def test_wide_diamond(self):
        g = build_graph(
            [
                ("a", "b"),
                ("a", "c"),
                ("a", "d"),
                ("b", "e"),
                ("c", "e"),
                ("d", "e"),
            ]
        )
        self._check_all_nodes(g, ["a", "b", "c", "d", "e"])

    def test_merge(self):
        g = build_graph(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("e", "f"),
                ("f", "c"),
            ]
        )
        self._check_all_nodes(g, ["a", "b", "c", "d", "e", "f"])

    def test_star(self):
        g = build_graph(
            [
                ("a", "b"),
                ("a", "c"),
                ("a", "d"),
                ("a", "e"),
            ]
        )
        self._check_all_nodes(g, ["a", "b", "c", "d", "e"])

    def test_fork_then_merge(self):
        g = build_graph(
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("a", "e"),
                ("e", "d"),
            ]
        )
        self._check_all_nodes(g, ["a", "b", "c", "d", "e"])

    def test_complex_dag(self):
        g = build_graph(
            [
                ("a", "b"),
                ("a", "c"),
                ("a", "d"),
                ("b", "e"),
                ("c", "e"),
                ("c", "f"),
                ("d", "f"),
                ("e", "g"),
                ("f", "g"),
            ]
        )
        self._check_all_nodes(g, ["a", "b", "c", "d", "e", "f", "g"])

    def test_two_components(self):
        g = build_graph([("a", "b"), ("c", "d"), ("e", "f")])
        self._check_all_nodes(g, ["a", "b", "c", "d", "e", "f"])

    def test_single_comparison(self):
        g = build_graph([("a", "b")])
        self._check_all_nodes(g, ["a", "b"])

    def test_long_chain(self):
        nodes = [str(i) for i in range(10)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(9)]
        g = build_graph(edges)
        self._check_all_nodes(g, nodes)

    def test_node_in_equally_long_chains(self):
        g = build_graph(
            [
                ("a", "b"),
                ("b", "c"),
                ("a", "d"),
                ("d", "c"),
            ]
        )
        node_c = g.get_node("c")
        main = node_c.get_chain(only_main=True)
        all_c = node_c.get_chain(only_main=False)
        assert len(main) == 1
        if all_c:
            main_len = main[0].length
            max_len = max(c.length for c in all_c)
            assert main_len == max_len, (
                f"Node c: main chain length {main_len} != "
                f"max chain length {max_len}"
            )


class TestChainLengthWithCycles:
    """Nodes in contradictory cycles must still get chain_length > 0."""

    def test_simple_cycle(self):
        cm = ChainManager()
        comparisons = [
            {"filename_a": "a.jpg", "filename_b": "b.jpg", "winner": "a.jpg"},
            {"filename_a": "b.jpg", "filename_b": "c.jpg", "winner": "b.jpg"},
            {"filename_a": "c.jpg", "filename_b": "a.jpg", "winner": "c.jpg"},
        ]
        cm.build(comparisons)
        for n in ("a.jpg", "b.jpg", "c.jpg"):
            assert (
                cm._chain_length.get(n, 0) > 0
            ), f"Node {n} in cycle has chain_length=0"

    def test_cycle_with_tail(self):
        """Nodes downstream of a cycle must also have positive chain_length."""
        cm = ChainManager()
        comparisons = [
            {"filename_a": "a.jpg", "filename_b": "b.jpg", "winner": "a.jpg"},
            {"filename_a": "b.jpg", "filename_b": "c.jpg", "winner": "b.jpg"},
            {"filename_a": "c.jpg", "filename_b": "a.jpg", "winner": "c.jpg"},
            {"filename_a": "c.jpg", "filename_b": "d.jpg", "winner": "c.jpg"},
        ]
        cm.build(comparisons)
        for n in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
            assert (
                cm._chain_length.get(n, 0) > 0
            ), f"Node {n} in/after cycle has chain_length=0"

    def test_no_false_positives_isolated(self):
        cm = ChainManager()
        comparisons: list[dict] = []
        cm.build(comparisons, all_filenames={"x.jpg", "y.jpg"})
        for n in ("x.jpg", "y.jpg"):
            assert (
                cm._chain_length.get(n, 0) == 0
            ), f"Isolated node {n} should have chain_length=0"
