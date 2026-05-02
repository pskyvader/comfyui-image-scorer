import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from shared.graph.crystal_graph import CrystalGraph


def test_build_empty_graph():
    """Test building a graph with no comparisons."""
    graph = CrystalGraph()
    graph.build_from_database(
        images=[{"filename": "img_a"}, {"filename": "img_b"}],
        comparisons=[],
    )

    # Isolated nodes form chains of length 1
    assert graph.get_max_height() == 1
    assert len(graph.get_top_nodes()) == 2
    assert len(graph.get_bottom_nodes()) == 2


def test_single_comparison():
    """Test graph with one comparison."""
    graph = CrystalGraph()
    graph.build_from_database(
        images=[{"filename": "img_a"}, {"filename": "img_b"}],
        comparisons=[
            {
                "filename_a": "img_a",
                "filename_b": "img_b",
                "winner": "img_a",
            }
        ],
    )

    info_a = graph.get_image_info("img_a")
    info_b = graph.get_image_info("img_b")

    # Chain: a->b, length 2. Both in chain, height 2.
    assert info_a["height"] == 2
    assert info_b["height"] == 2
    assert not info_a["better"]  # a has no betters
    assert "img_a" in info_b["better"]  # b's better is a
    assert "img_b" in info_a["worse"]


def test_multiple_comparisons_chain():
    """Test a chain: img_a beats img_b, img_b beats img_c."""
    graph = CrystalGraph()
    graph.build_from_database(
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

    info_a = graph.get_image_info("img_a")
    info_b = graph.get_image_info("img_b")
    info_c = graph.get_image_info("img_c")

    # Chain a->b->c, length 3. All height 3.
    assert info_a["height"] == 3
    assert info_b["height"] == 3
    assert info_c["height"] == 3


def test_get_images_by_height():
    """Test retrieving images at a specific height."""
    graph = CrystalGraph()
    graph.build_from_database(
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

    # All in chain of length 3, so height 3
    assert graph.get_images_by_height(3) == ["img_a", "img_b", "img_c"]


def test_top_and_bottom_nodes():
    """Test identification of top and bottom nodes."""
    graph = CrystalGraph()
    graph.build_from_database(
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

    top_nodes = graph.get_top_nodes()
    bottom_nodes = graph.get_bottom_nodes()

    assert "img_a" in top_nodes
    assert "img_c" in top_nodes  # img_c never compared, so no betters
    assert "img_b" in bottom_nodes
    assert "img_c" in bottom_nodes


def test_graph_stats():
    """Test graph statistics generation."""
    graph = CrystalGraph()
    graph.build_from_database(
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
    assert stats["max_height"] == 2


def test_get_all_heights():
    """Test getting all height levels."""
    graph = CrystalGraph()
    graph.build_from_database(
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

    all_heights = graph.get_all_heights()
    assert 3 in all_heights
    assert len(all_heights[3]) == 3


def test_unknown_image_info():
    """Test getting info for unknown image returns None."""
    graph = CrystalGraph()
    graph.build_from_database(
        images=[{"filename": "img_a"}],
        comparisons=[],
    )

    info = graph.get_image_info("nonexistent")
    assert info is None


def test_are_in_same_path_directed():
    """Test are_in_same_path with directed edges."""
    graph = CrystalGraph()
    graph.build_from_database(
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
    graph.build_from_database(
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
    graph.build_from_database(
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
    graph.build_from_database(
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

    info_c = graph.get_image_info("img_c")
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
    graph.build_from_database(images=images, comparisons=comparisons)

    assert graph.get_image_info("c")["height"] == 6
    assert graph.get_image_info("g")["height"] == 4
    assert graph.get_image_info("f")["height"] == 6
    assert graph.get_image_info("a")["height"] == 6
    assert graph.get_image_info("b")["height"] == 6
    assert graph.get_image_info("d")["height"] == 6
    assert graph.get_image_info("e")["height"] == 6


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
    graph.build_from_database(
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
    graph2.build_from_database(
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
    graph.build_from_database(images=images, comparisons=comparisons)

    # Should have 3 components: [a, b], [c, d], [e]
    assert len(graph.chain_members_by_component) == 3

    # Check that a and b are in the same component
    comp_a = graph.component_by_filename.get("a")
    comp_b = graph.component_by_filename.get("b")
    assert comp_a == comp_b

    # Check that c and d are in the same component
    comp_c = graph.component_by_filename.get("c")
    comp_d = graph.component_by_filename.get("d")
    assert comp_c == comp_d

    # Check that a and c are in different components
    assert comp_a != comp_c

    # Check that e is in its own component
    comp_e = graph.component_by_filename.get("e")
    assert comp_e is not None
    assert len(graph.chain_members_by_component[comp_e]) == 1
