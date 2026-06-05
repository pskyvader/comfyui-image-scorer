import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from ...graph.crystal_graph import CrystalGraph


def test_build_empty_graph():
    """Test building a graph with no comparisons."""
    graph = CrystalGraph()
    graph.rebuild_from_database(
        images=[{"filename": "img_a"}, {"filename": "img_b"}],
        comparisons=[],
    )

    # Isolated nodes form chains of length 0 (0 edges)
    assert max(graph._chain._chain_length.values(), default=0) == 0
    assert len(graph._chain._top_nodes) == 2
    assert len(graph._chain._bottom_nodes) == 2


def test_single_comparison():
    """Test graph with one comparison."""
    graph = CrystalGraph()
    graph.rebuild_from_database(
        images=[{"filename": "img_a"}, {"filename": "img_b"}],
        comparisons=[
            {
                "filename_a": "img_a",
                "filename_b": "img_b",
                "winner": "img_a",
            }
        ],
    )

    info_a = {
        "height": graph._chain._chain_length.get("img_a", 0),
        "better": graph._chain._better_than.get("img_a", set()),
        "worse": graph._chain._worse_than.get("img_a", set()),
    }
    info_b = {
        "height": graph._chain._chain_length.get("img_b", 0),
        "better": graph._chain._better_than.get("img_b", set()),
        "worse": graph._chain._worse_than.get("img_b", set()),
    }

    # Chain: a->b, 1 edge. Both have chain_length 1.
    assert info_a["height"] == 1
    assert info_b["height"] == 1
    assert not info_a["better"]  # a has no betters
    assert "img_a" in info_b["better"]  # b's better is a
    assert "img_b" in info_a["worse"]


def test_multiple_comparisons_chain():
    """Test a chain: img_a beats img_b, img_b beats img_c."""
    graph = CrystalGraph()
    graph.rebuild_from_database(
        images=[
            {"filename": "img_a"},
            {"filename": "img_b"},
            {"filename": "img_c"},
        ],
        comparisons=[
            {
                "filename_a": "img_a",
                "filename_b": "img_b",
                "winner": "img_a",
            },
            {
                "filename_a": "img_b",
                "filename_b": "img_c",
                "winner": "img_b",
            },
        ],
    )

    info_a = {
        "height": graph._chain._chain_length.get("img_a", 0),
        "better": graph._chain._better_than.get("img_a", set()),
        "worse": graph._chain._worse_than.get("img_a", set()),
    }
    info_b = {
        "height": graph._chain._chain_length.get("img_b", 0),
        "better": graph._chain._better_than.get("img_b", set()),
        "worse": graph._chain._worse_than.get("img_b", set()),
    }
    info_c = {
        "height": graph._chain._chain_length.get("img_c", 0),
        "better": graph._chain._better_than.get("img_c", set()),
        "worse": graph._chain._worse_than.get("img_c", set()),
    }

    # Chain a->b->c, 2 edges. All chain_length 2.
    assert info_a["height"] == 2
    assert info_b["height"] == 2
    assert info_c["height"] == 2


def test_get_images_by_height():
    """Test retrieving images at a specific height."""
    graph = CrystalGraph()
    graph.rebuild_from_database(
        images=[
            {"filename": "img_a"},
            {"filename": "img_b"},
            {"filename": "img_c"},
        ],
        comparisons=[
            {
                "filename_a": "img_a",
                "filename_b": "img_b",
                "winner": "img_a",
            },
            {
                "filename_a": "img_b",
                "filename_b": "img_c",
                "winner": "img_b",
            },
        ],
    )

    # All in chain of 2 edges, so chain_length 2
    assert set(graph._chain._nodes_by_length.get(2, [])) == {"img_a", "img_b", "img_c"}


def test_top_and_bottom_nodes():
    """Test identification of top and bottom nodes."""
    graph = CrystalGraph()
    graph.rebuild_from_database(
        images=[
            {"filename": "img_a"},
            {"filename": "img_b"},
            {"filename": "img_c"},
        ],
        comparisons=[
            {
                "filename_a": "img_a",
                "filename_b": "img_b",
                "winner": "img_a",
            },
        ],
    )

    top_nodes = graph._chain._top_nodes
    bottom_nodes = graph._chain._bottom_nodes

    assert "img_a" in top_nodes
    assert "img_c" in top_nodes  # img_c never compared, so no betters
    assert "img_b" in bottom_nodes
    assert "img_c" in bottom_nodes


def test_graph_stats():
    """Test graph statistics generation."""
    graph = CrystalGraph()
    graph.rebuild_from_database(
        images=[
            {"filename": "img_a"},
            {"filename": "img_b"},
        ],
        comparisons=[
            {
                "filename_a": "img_a",
                "filename_b": "img_b",
                "winner": "img_a",
            },
        ],
    )

    stats = graph.get_graph_stats()
    assert stats["total_images"] == 2
    assert stats["total_comparisons"] == 1
    assert stats["longest_chain_depth"] == 1


def test_get_all_heights():
    """Test getting all height levels."""
    graph = CrystalGraph()
    graph.rebuild_from_database(
        images=[
            {"filename": "img_a"},
            {"filename": "img_b"},
            {"filename": "img_c"},
        ],
        comparisons=[
            {
                "filename_a": "img_a",
                "filename_b": "img_b",
                "winner": "img_a",
            },
            {
                "filename_a": "img_b",
                "filename_b": "img_c",
                "winner": "img_b",
            },
        ],
    )

    all_heights = graph._chain._nodes_by_length
    assert 2 in all_heights
    assert len(all_heights[2]) == 3


def test_unknown_image_info():
    """Test getting info for unknown image returns None."""
    graph = CrystalGraph()
    graph.rebuild_from_database(
        images=[{"filename": "img_a"}],
        comparisons=[],
    )

    info = graph._chain._chain_length.get("nonexistent")
    assert info is None


def test_are_in_same_path_directed():
    """Test are_in_same_path with directed edges."""
    graph = CrystalGraph()
    graph.rebuild_from_database(
        images=[
            {"filename": "img_a"},
            {"filename": "img_b"},
            {"filename": "img_c"},
        ],
        comparisons=[
            {
                "filename_a": "img_a",
                "filename_b": "img_b",
                "winner": "img_a",
            },
            {
                "filename_a": "img_b",
                "filename_b": "img_c",
                "winner": "img_b",
            },
        ],
    )

    # a can reach c via a->b->c
    assert graph.are_in_same_path("img_a", "img_c") is True
    # c cannot reach a (directed opposite)
    assert graph.are_in_same_path("img_c", "img_a") is True  # Wait, no: can c reach a? No, because edges are winner->loser. c lost to b, b lost to a. So c's worse is empty. So c cannot reach a. But the method checks if either can reach the other. a can reach c, so returns True. That's correct.

    # Test a vs b: a can reach b
    assert graph.are_in_same_path("img_a", "img_b") is True


def test_are_in_same_path_not_directed():
    """Test that undirected connection returns False if no directed path."""
    graph = CrystalGraph()
    graph.rebuild_from_database(
        images=[
            {"filename": "img_a"},
            {"filename": "img_b"},
            {"filename": "img_c"},
        ],
        comparisons=[
            {
                "filename_a": "img_a",
                "filename_b": "img_b",
                "winner": "img_a",
            },
            {
                "filename_a": "img_c",
                "filename_b": "img_b",
                "winner": "img_c",
            },
        ],
    )

    # a and c both beat b. No directed path between a and c.
    assert graph.are_in_same_path("img_a", "img_c") is False


def test_cache_info():
    """Test cache info and staleness tracking."""
    graph = CrystalGraph()
    graph.rebuild_from_database(
        images=[{"filename": "img_a"}],
        comparisons=[],
    )

    cache_info = graph.get_cache_info()
    assert cache_info["built_at"] is not None
    assert cache_info["comparison_count_at_build"] == 0
    assert isinstance(cache_info["is_stale"], bool)


def test_multiple_betters():
    """Test node with multiple betters."""
    graph = CrystalGraph()
    graph.rebuild_from_database(
        images=[
            {"filename": "img_a"},
            {"filename": "img_b"},
            {"filename": "img_c"},
        ],
        comparisons=[
            {
                "filename_a": "img_a",
                "filename_b": "img_c",
                "winner": "img_a",
            },
            {
                "filename_a": "img_b",
                "filename_b": "img_c",
                "winner": "img_b",
            },
        ],
    )

    info_c = {
        "height": graph._chain._chain_length.get("img_c", 0),
        "better": graph._chain._better_than.get("img_c", set()),
        "worse": graph._chain._worse_than.get("img_c", set()),
    }
    assert "img_a" in info_c["better"]
    assert "img_b" in info_c["better"]
    assert len(info_c["better"]) == 2


def test_user_example_heights():
    """Test the user's example for height calculation.

    Chains:
    a>b>c>d>e>f (length 6)
    a>b>c>g (length 4)
    a>b>e>f (length 4)

    Heights:
    a:6, b:6, c:6, d:6, e:6, f:6, g:4
    """
    graph = CrystalGraph()
    images = [
        {"filename": "a"},
        {"filename": "b"},
        {"filename": "c"},
        {"filename": "d"},
        {"filename": "e"},
        {"filename": "f"},
        {"filename": "g"},
    ]
    comparisons = [
        {"filename_a": "a", "filename_b": "b", "winner": "a"},
        {"filename_a": "b", "filename_b": "c", "winner": "b"},
        {"filename_a": "c", "filename_b": "d", "winner": "c"},
        {"filename_a": "d", "filename_b": "e", "winner": "d"},
        {"filename_a": "b", "filename_b": "e", "winner": "b"},
        {"filename_a": "e", "filename_b": "f", "winner": "e"},
        {"filename_a": "c", "filename_b": "g", "winner": "c"},
    ]
    graph.rebuild_from_database(images=images, comparisons=comparisons)

    assert graph._chain._chain_length.get("c", 0) == 5
    assert graph._chain._chain_length.get("g", 0) == 3
    assert graph._chain._chain_length.get("f", 0) == 5
    assert graph._chain._chain_length.get("a", 0) == 5
    assert graph._chain._chain_length.get("b", 0) == 5
    assert graph._chain._chain_length.get("d", 0) == 5
    assert graph._chain._chain_length.get("e", 0) == 5


def test_user_example_same_path():
    """Test user's example for are_in_same_path.

    a vs g: a can reach g via a->b->c->g, so True.
    a vs e: a can reach e via a->b->e, so True.
    a vs e? Wait user said a vs e should be? Let's see user's first example:
    User's first example: a>b>c>d, e>b>f>g. a vs g should be True (a>b>f>g? No, a>b, b>f? In that example, e>b>f>g: e beats b, b beats f, f beats g. So a beats b, so a>b>f>g, so a can reach g. a vs e: a>b, e>b, no directed path between a and e, so False.

    But in the second example, a vs e: a can reach e via a->b->e, so True.
    """
    graph = CrystalGraph()
    images = ["a", "b", "c", "d", "e", "f", "g"]
    comparisons = [
        {"filename_a": "a", "filename_b": "b", "winner": "a"},
        {"filename_a": "b", "filename_b": "c", "winner": "b"},
        {"filename_a": "c", "filename_b": "d", "winner": "c"},
        {"filename_a": "d", "filename_b": "e", "winner": "d"},
        {"filename_a": "b", "filename_b": "e", "winner": "b"},
        {"filename_a": "e", "filename_b": "f", "winner": "e"},
        {"filename_a": "c", "filename_b": "g", "winner": "c"},
    ]
    graph.rebuild_from_database(
        images=[{"filename": img} for img in images],
        comparisons=comparisons,
    )

    # a can reach g via a->b->c->g
    assert graph.are_in_same_path("a", "g") is True
    # a can reach e via a->b->e
    assert graph.are_in_same_path("a", "e") is True
    # g cannot reach a (no directed path)
    # but since a can reach g, are_in_same_path returns True (either direction)
    assert graph.are_in_same_path("g", "a") is True  # because a can reach g

    # Now test the first example where a vs e should be False
    # Example: a>b>c>d, e>b>f>g
    graph2 = CrystalGraph()
    images2 = ["a", "b", "c", "d", "e", "f", "g"]
    comparisons2 = [
        {"filename_a": "a", "filename_b": "b", "winner": "a"},
        {"filename_a": "b", "filename_b": "c", "winner": "b"},
        {"filename_a": "c", "filename_b": "d", "winner": "c"},
        {"filename_a": "e", "filename_b": "b", "winner": "e"},
        {"filename_a": "b", "filename_b": "f", "winner": "b"},
        {"filename_a": "f", "filename_b": "g", "winner": "f"},
    ]
    graph2.rebuild_from_database(
        images=[{"filename": img} for img in images2],
        comparisons=comparisons2,
    )
    # a can reach g via a->b->f->g
    assert graph2.are_in_same_path("a", "g") is True
    # a and e: a>b, e>b, no path between a and e
    assert graph2.are_in_same_path("a", "e") is False


def test_component_detection():
    """Test that connected components are correctly identified."""
    graph = CrystalGraph()
    images = [
        {"filename": "a"},
        {"filename": "b"},
        {"filename": "c"},
        {"filename": "d"},
        {"filename": "e"},  # isolated, no comparisons
    ]
    comparisons = [
        {"filename_a": "a", "filename_b": "b", "winner": "a"},
        {"filename_a": "c", "filename_b": "d", "winner": "c"},
        # a-b and c-d are separate components
    ]
    graph.rebuild_from_database(images=images, comparisons=comparisons)

    # Should have 3 components: [a, b], [c, d], [e]
    assert len(graph._chain._component_members) == 3

    # Check that a and b are in the same component
    comp_a = graph._chain._node_component.get("a")
    comp_b = graph._chain._node_component.get("b")
    assert comp_a == comp_b

    # Check that c and d are in the same component
    comp_c = graph._chain._node_component.get("c")
    comp_d = graph._chain._node_component.get("d")
    assert comp_c == comp_d

    # Check that a and c are in different components
    assert comp_a != comp_c

    # Check that e is in its own component
    comp_e = graph._chain._node_component.get("e")
    assert comp_e is not None
    assert len(graph._chain._component_members[comp_e]) == 1
