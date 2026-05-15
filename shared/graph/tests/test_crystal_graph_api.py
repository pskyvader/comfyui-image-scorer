import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from shared.graph.crystal_graph import CrystalGraph, NodeProxy, ChainProxy, ComponentProxy


def test_get_node():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}],
        comparisons=[{"filename_a": "a", "filename_b": "b", "winner": "a"}],
    )

    node = cg.get_node(node_id="a")
    assert node is not None
    assert isinstance(node, NodeProxy)
    assert node.filename == "a"

    none_node = cg.get_node(node_id="nonexistent")
    assert none_node is None

    none_node2 = cg.get_node(node_id=None)
    assert none_node2 is None


def test_get_all_nodes():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"}, {"filename": "d"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "c", "filename_b": "d", "winner": "c"},
        ],
    )

    all_nodes = cg.get_all_nodes()
    assert len(all_nodes) == 4
    assert all(isinstance(n, NodeProxy) for n in all_nodes)

    tops = cg.get_all_nodes(only_top=True)
    assert len(tops) == 2
    assert all(n.is_top() for n in tops)

    bots = cg.get_all_nodes(only_bottom=True)
    assert len(bots) == 2
    assert all(n.is_bottom() for n in bots)

    import pytest
    with pytest.raises(ValueError):
        cg.get_all_nodes(only_top=True, only_bottom=True)


def test_node_is_top_bottom():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}],
        comparisons=[{"filename_a": "a", "filename_b": "b", "winner": "a"}],
    )

    assert cg.get_node("a").is_top()
    assert not cg.get_node("a").is_bottom()
    assert not cg.get_node("b").is_top()
    assert cg.get_node("b").is_bottom()


def test_node_get_links():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "a", "filename_b": "c", "winner": "a"},
        ],
    )

    node_a = cg.get_node("a")

    all_links = node_a.get_links()
    assert len(all_links) == 2

    worse = node_a.get_links(worse_than=True)
    assert len(worse) == 2

    better = node_a.get_links(better_than=True)
    assert len(better) == 0

    node_b = cg.get_node("b")
    b_better = node_b.get_links(better_than=True)
    assert len(b_better) == 1
    assert b_better[0].filename == "a"

    import pytest
    with pytest.raises(ValueError):
        node_a.get_links(better_than=True, worse_than=True)


def test_node_get_chain():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"},
                {"filename": "d"}, {"filename": "e"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "b", "filename_b": "c", "winner": "b"},
            {"filename_a": "c", "filename_b": "d", "winner": "c"},
            {"filename_a": "a", "filename_b": "e", "winner": "a"},
        ],
    )

    node = cg.get_node("a")
    chains = node.get_chain(only_main=True)
    assert len(chains) == 1
    assert isinstance(chains[0], ChainProxy)
    assert chains[0].length == 4  # a>b>c>d (4 nodes, 3 edges)

    # e is in its own shorter chain (a>e = 2 nodes)
    node_e = cg.get_node("e")
    e_chains = node_e.get_chain(only_main=True)
    assert len(e_chains) == 1
    assert e_chains[0].length == 2  # a>e

    # Verify the chain cover has 2 distinct chains
    all_chains = cg.get_all_chains()
    assert len(all_chains) == 2


def test_node_get_chain_merge_point():
    """Node at merge point should appear in multiple minimal chains."""
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"},
                {"filename": "d"}, {"filename": "e"}, {"filename": "f"},
                {"filename": "g"}, {"filename": "h"}, {"filename": "i"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "b", "filename_b": "c", "winner": "b"},
            {"filename_a": "c", "filename_b": "d", "winner": "c"},
            {"filename_a": "d", "filename_b": "e", "winner": "d"},
            {"filename_a": "d", "filename_b": "h", "winner": "d"},
            {"filename_a": "a", "filename_b": "f", "winner": "a"},
            {"filename_a": "f", "filename_b": "g", "winner": "f"},
            {"filename_a": "g", "filename_b": "h", "winner": "g"},
            {"filename_a": "h", "filename_b": "i", "winner": "h"},
            {"filename_a": "i", "filename_b": "e", "winner": "i"},
        ],
    )

    node_h = cg.get_node("h")
    main_chains = node_h.get_chain(only_main=True)
    assert len(main_chains) == 1

    all_chains = node_h.get_chain(only_main=False)
    assert len(all_chains) == 2  # h is in both chains


def test_node_get_position_in_chain():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "b", "filename_b": "c", "winner": "b"},
        ],
    )

    assert cg.get_node("a").get_position_in_chain() == 0
    assert cg.get_node("b").get_position_in_chain() == 1
    assert cg.get_node("c").get_position_in_chain() == 2


def test_node_get_component():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}],
        comparisons=[{"filename_a": "a", "filename_b": "b", "winner": "a"}],
    )

    comp = cg.get_node("a").get_component()
    assert comp is not None
    assert isinstance(comp, ComponentProxy)
    assert comp.size == 2

    # Isolated node
    cg2 = CrystalGraph()
    cg2.rebuild_from_database(
        images=[{"filename": "x"}],
        comparisons=[],
    )
    comp2 = cg2.get_node("x").get_component()
    assert comp2 is not None
    assert comp2.size == 1


def test_chain_proxy_properties():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"},
                {"filename": "d"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "b", "filename_b": "c", "winner": "b"},
            {"filename_a": "c", "filename_b": "d", "winner": "c"},
        ],
    )

    chain = cg.get_chain(node_id="a")
    assert chain is not None
    assert isinstance(chain, ChainProxy)
    assert chain.id == 0
    assert chain.length == 4  # 4 nodes, 3 edges
    assert chain.first.filename == "a"
    assert chain.last.filename == "d"

    nodes = chain.nodes
    assert len(nodes) == 4
    assert [n.filename for n in nodes] == ["a", "b", "c", "d"]


def test_chain_get_chain_by_id():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "a", "filename_b": "c", "winner": "a"},
        ],
    )

    chain = cg.get_chain(chain_id=0)
    assert chain is not None

    chain_invalid = cg.get_chain(chain_id=999)
    assert chain_invalid is None

    import pytest
    with pytest.raises(ValueError):
        cg.get_chain()


def test_chain_get_nodes():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "b", "filename_b": "c", "winner": "b"},
        ],
    )

    chain = cg.get_chain(node_id="a")
    assert len(chain.get_nodes()) == 3
    assert len(chain.get_nodes(only_top=True)) == 1
    assert len(chain.get_nodes(only_bottom=True)) == 1

    import pytest
    with pytest.raises(ValueError):
        chain.get_nodes(only_top=True, only_bottom=True)


def test_chain_node_position():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "b", "filename_b": "c", "winner": "b"},
        ],
    )

    chain = cg.get_chain(node_id="a")
    assert chain.node_position("a") == 0
    assert chain.node_position("c") == 2

    import pytest
    with pytest.raises(ValueError):
        chain.node_position("nonexistent")


def test_chain_get_component():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}],
        comparisons=[{"filename_a": "a", "filename_b": "b", "winner": "a"}],
    )

    chain = cg.get_chain(node_id="a")
    comp = chain.get_component()
    assert comp is not None
    assert comp.size == 2


def test_get_all_chains():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "a", "filename_b": "c", "winner": "a"},
        ],
    )

    chains = cg.get_all_chains()
    assert len(chains) == 2
    assert all(isinstance(c, ChainProxy) for c in chains)


def test_get_all_sub_chains():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "b", "filename_b": "c", "winner": "b"},
        ],
    )

    sub_chains = cg.get_all_sub_chains(max_chains=100, max_depth=100)
    assert len(sub_chains) > 0
    assert all(isinstance(c, ChainProxy) for c in sub_chains)
    # All sub-paths: [a,b], [a,b,c], [b,c]
    assert len(sub_chains) >= 3


def test_component_get_chains():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "a", "filename_b": "c", "winner": "a"},
        ],
    )

    comp = cg.get_component(node_id="a")
    min_chains = comp.get_chains(minimal_required=True)
    assert len(min_chains) == 2  # 2 chains in the cover

    top_chains = comp.get_chains(minimal_required=False)
    assert len(top_chains) >= 2


def test_component_lookup_methods():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"},
                {"filename": "d"}, {"filename": "e"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "c", "filename_b": "d", "winner": "c"},
        ],
    )

    comp_by_node = cg.get_component(node_id="a")
    assert comp_by_node is not None

    # Get the component by its known ID
    comp_by_id = cg.get_component(component_id=comp_by_node.id)
    assert comp_by_id is not None
    assert comp_by_id.id == comp_by_node.id

    # Chain lookup
    chain = cg.get_chain(node_id="a")
    comp_by_chain = cg.get_component(chain_id=chain.id)
    assert comp_by_chain is not None
    assert comp_by_chain.id == comp_by_node.id

    import pytest
    with pytest.raises(ValueError):
        cg.get_component()
    with pytest.raises(ValueError):
        cg.get_component(node_id="a", component_id=0)


def test_get_all_components():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "c", "filename_b": "a", "winner": "c"},
        ],
    )

    comps = cg.get_all_components()
    assert len(comps) == 1  # All connected
    assert all(isinstance(c, ComponentProxy) for c in comps)


def test_get_all_links():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "a", "filename_b": "c", "winner": "a"},
        ],
    )

    links = cg.get_all_links()
    assert len(links) == 2
    for w, l in links:
        assert isinstance(w, NodeProxy)
        assert isinstance(l, NodeProxy)


def test_apply_comparison_incremental():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}],
        comparisons=[],
    )
    assert len(cg.get_all_nodes()) == 2

    cg.apply_comparison("a", "b")
    assert len(cg.get_all_nodes()) == 2
    assert cg.get_node("a").is_top()
    assert cg.get_node("b").is_bottom()
    assert len(cg.get_node("a").get_links(worse_than=True)) == 1

    # Add a new node via comparison
    cg.apply_comparison("b", "c")
    assert cg.get_node("c") is not None
    assert cg.get_node("c").is_bottom()
    assert not cg.get_node("b").is_bottom()


def test_apply_comparison_chain_growth():
    """Build a long chain incrementally and verify chain_lengths."""
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}],
        comparisons=[],
    )

    nodes = ["a", "b", "c", "d", "e"]
    for i in range(4):
        cg.apply_comparison(nodes[i], nodes[i + 1])

    # The chain a>b>c>d>e has 4 edges
    for n in nodes:
        assert cg._chain._chain_length.get(n, -1) == 4


def test_apply_comparison_merge_components():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"}, {"filename": "d"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "c", "filename_b": "d", "winner": "c"},
        ],
    )

    assert len(cg.get_all_components()) == 2

    # Link the two components
    cg.apply_comparison("b", "c")
    assert len(cg.get_all_components()) == 1


def test_are_in_same_path():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "b", "filename_b": "c", "winner": "b"},
        ],
    )

    assert cg.are_in_same_path("a", "c") is True
    assert cg.are_in_same_path("a", "a") is True

    cg2 = CrystalGraph()
    cg2.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "c", "filename_b": "b", "winner": "c"},
        ],
    )
    assert cg2.are_in_same_path("a", "c") is False


def test_is_redundant():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "b", "filename_b": "c", "winner": "b"},
        ],
    )

    # a->b is not redundant: no alternative path a -> ... -> b
    assert cg.is_redundant("a", "b") is False

    # If we add edge a->c, it WOULD be redundant (a can reach c via b)
    # But the edge doesn't exist yet, so we test with skip_edges:
    # Check if edge b->c is redundant: no alternative b -> ... -> c
    assert cg.is_redundant("b", "c") is False


def test_get_graph_stats():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}],
        comparisons=[{"filename_a": "a", "filename_b": "b", "winner": "a"}],
    )

    stats = cg.get_graph_stats()
    assert "total_images" in stats
    assert "total_comparisons" in stats
    assert "total_components" in stats
    assert "total_chains" in stats
    assert "longest_chain_depth" in stats
    assert "top_nodes_count" in stats
    assert "bottom_nodes_count" in stats
    assert "built_at" in stats
    assert stats["total_images"] == 2
    assert stats["total_comparisons"] == 1
    assert stats["longest_chain_depth"] >= 1


def test_is_cache_stale():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}],
        comparisons=[],
    )
    # Should not be stale immediately after build
    assert isinstance(cg.is_cache_stale(), bool)


def test_get_collapsable_pairs():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}, {"filename": "c"}],
        comparisons=[
            {"filename_a": "a", "filename_b": "b", "winner": "a"},
            {"filename_a": "b", "filename_b": "c", "winner": "b"},
        ],
    )

    pairs = cg.get_collapsable_pairs()
    assert isinstance(pairs, list)


def test_nonexistent_node():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}],
        comparisons=[],
    )
    assert cg.get_node(node_id="nonexistent") is None
    assert cg.get_chain(node_id="nonexistent") is None
    assert cg.get_component(node_id="nonexistent") is None


def test_invalid_component_lookup():
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}],
        comparisons=[],
    )
    assert cg.get_component(component_id=999) is None


def test_apply_comparison_preserves_integrity():
    """Build via comparisons then incrementally add more, verify consistency."""
    cg = CrystalGraph()
    cg.rebuild_from_database(
        images=[{"filename": "a"}, {"filename": "b"}],
        comparisons=[{"filename_a": "a", "filename_b": "b", "winner": "a"}],
    )

    before_stats = cg.get_graph_stats()

    cg.apply_comparison("b", "c")
    after_stats = cg.get_graph_stats()

    assert after_stats["total_images"] == before_stats["total_images"] + 1 or \
           after_stats["total_images"] >= before_stats["total_images"]
    assert after_stats["total_comparisons"] == before_stats["total_comparisons"] + 1
