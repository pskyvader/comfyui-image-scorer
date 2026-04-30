import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from shared.graph import enumerate_simple_chains, get_current_connections


def test_get_current_connections_builds_directional_and_chain_indexes():
    images = [
        {"filename": "img_a"},
        {"filename": "img_b"},
        {"filename": "img_c"},
        {"filename": "img_loose"},
    ]
    comparisons = [
        {
            "filename_a": "img_a",
            "filename_b": "img_b",
            "winner": "img_a",
            "weight": 1.0,
            "transitive_depth": 0,
        },
        {
            "filename_a": "img_b",
            "filename_b": "img_c",
            "winner": "img_b",
            "weight": 1.0,
            "transitive_depth": 0,
        },
    ]

    connections = get_current_connections(
        all_images=images,
        all_comparisons=comparisons,
    )

    assert connections.compared_pairs == {
        ("img_a", "img_b"),
        ("img_b", "img_c"),
    }
    assert connections.winners_by_image["img_a"] == [("img_b", 1.0)]
    assert connections.winners_by_image["img_b"] == [("img_c", 1.0)]
    assert connections.losers_by_image["img_b"] == [("img_a", 1.0)]
    assert connections.graph_nodes == {"img_a", "img_b", "img_c"}
    assert connections.chain_length_by_filename["img_a"] == 2
    assert connections.chain_length_by_filename["img_b"] == 2
    assert connections.chain_length_by_filename["img_c"] == 2
    assert connections.chain_length_by_filename["img_loose"] == 0


def test_enumerate_simple_chains_uses_shared_connections():
    images = [
        {"filename": "img_a"},
        {"filename": "img_b"},
        {"filename": "img_c"},
    ]
    comparisons = [
        {
            "filename_a": "img_a",
            "filename_b": "img_b",
            "winner": "img_a",
            "weight": 1.0,
            "transitive_depth": 0,
        },
        {
            "filename_a": "img_b",
            "filename_b": "img_c",
            "winner": "img_b",
            "weight": 1.0,
            "transitive_depth": 0,
        },
    ]

    connections = get_current_connections(
        all_images=images,
        all_comparisons=comparisons,
    )
    chains, chain_lengths = enumerate_simple_chains(
        connections,
        max_chain_depth=10,
        max_chains=100,
    )

    assert ["img_a", "img_b"] in chains
    assert ["img_b", "img_c"] in chains
    assert ["img_a", "img_b", "img_c"] in chains
    assert chain_lengths[2] == 2
    assert chain_lengths[3] == 1
