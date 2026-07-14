"""Check that bottom nodes are the last element in their main chain.

We enforce strict validation of the top/bottom requirements and include a
performance test for large linear chains that scales via DATASET_SIZE.
"""

import logging
import time
import pytest
from tqdm import tqdm
from comfyui_image_scorer.shared.graph.crystal_graph import CrystalGraph
from comfyui_image_scorer.shared.graph.chain_manager import ChainManager
from comfyui_image_scorer.external_modules.database_structure.comparisons_table import (
    get_images_with_only_wins,
    get_images_with_only_losses,
)

logger = logging.getLogger(__name__)

# Change this variable to test the 30-second time limit on large chains.
# Set to 100 by default. To stress test performance limits, try 10000 or 35000.
DATASET_SIZE = 100000


def test_bottom_nodes_are_chain_last() -> None:
    """Strictly assert that chains always start at tops and end at bottoms."""
    logger.debug("Starting test_bottom_nodes_are_chain_last...")

    # Isolate test by creating a fresh CrystalGraph and loading DB
    cg = CrystalGraph()
    cg.rebuild_from_database()

    bad = 0
    total = 0
    for length, chains_dict in cg.get_chains_map().items():
        logger.debug(f"Checking {len(chains_dict)} chains of length {length}")
        for chain_id, (chain_proxy, chain_nodes) in chains_dict.items():
            if not chain_nodes:
                continue
            total += 1
            first = chain_nodes[0][0]
            last = chain_nodes[-1][0]

            if not first.is_top():
                logger.error(f"Chain ends at {first.name} which is NOT a top node.")
                bad += 1
            if not last.is_bottom():
                logger.error(f"Chain ends at {last.name} which is NOT a bottom node.")
                bad += 1

    assert bad == 0, f"{bad}/{total} chains do not start/end at the absolute extremes!"


def test_performance_on_large_chains() -> None:
    """Test that ChainManager processes a large dataset under 30 seconds."""
    logger.debug("Starting test_performance_on_large_chains...")
    cm = ChainManager()

    from comfyui_image_scorer.external_modules.database_structure.comparisons_table import (
        get_all_comparisons,
    )

    all_real_comparisons = get_all_comparisons()

    if not all_real_comparisons:
        pytest.skip("No real records found in the database to test performance.")

    unique_images = set()
    for comp in all_real_comparisons:
        unique_images.add(comp["filename_a"])
        unique_images.add(comp["filename_b"])

    # Get up to DATASET_SIZE images
    selected_images = set(list(unique_images)[:DATASET_SIZE])
    logger.debug(f"Selected {len(selected_images)} unique images for the subset.")

    comparisons = []
    with tqdm(all_real_comparisons, desc="TEST: Filtering comparisons") as pbar:
        for c in pbar:
            if c["filename_a"] in selected_images or c["filename_b"] in selected_images:
                comparisons.append(c)
    logger.debug(f"Filtered down to {len(comparisons)} comparisons.")

    start_time = time.perf_counter()
    cm.build(comparisons)
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    logger.info(
        f"ChainManager.build processed {len(comparisons)} comparisons in {elapsed:.4f} seconds."
    )

    # Requirement 3: Must never take longer than 30 seconds
    assert (
        elapsed < 30.0
    ), f"Processing took {elapsed:.2f}s, which exceeds the 30s limit!"


def test_cycles_do_not_prevent_bottom_reachability() -> None:
    """Test that cyclic paths still properly reach and end at the absolute bottom."""
    logger.debug("Starting test_cycles_do_not_prevent_bottom_reachability...")
    cm = ChainManager()

    comparisons = [
        {"filename_a": "a", "filename_b": "b", "winner": "a"},
        {"filename_a": "b", "filename_b": "c", "winner": "b"},
        {"filename_a": "c", "filename_b": "a", "winner": "c"},  # cycle a>b>c>a
        {
            "filename_a": "a",
            "filename_b": "d",
            "winner": "a",
        },  # branch to absolute bottom d
    ]

    cm.build(comparisons)

    chains = cm.get_chains()
    for cid, chain in chains.items():
        # Every chain built MUST end at 'd', because 'd' is the only absolute bottom!
        assert (
            chain[-1] == "d"
        ), f"Chain {chain} ends at {chain[-1]} instead of the absolute bottom 'd'"


def test_transitive_reduction_sorting() -> None:
    """Test that a>b, b>c, a>c correctly builds a single sorted chain a>b>c."""
    logger.debug("Starting test_transitive_reduction_sorting...")
    cm = ChainManager()

    comparisons = [
        {"filename_a": "a", "filename_b": "b", "winner": "a"},
        {"filename_a": "b", "filename_b": "c", "winner": "b"},
        {"filename_a": "a", "filename_b": "c", "winner": "a"},  # Transitive edge
    ]

    cm.build(comparisons)
    chains = cm.get_chains()

    # Requirement 2: should return just 1 main chain a>b>c
    assert len(chains) == 1, f"Expected 1 chain, got {len(chains)}"
    main_chain = list(chains.values())[0]
    assert main_chain == ["a", "b", "c"], f"Expected ['a', 'b', 'c'], got {main_chain}"


def test_uncompared_nodes_are_isolated_top_bottom() -> None:
    """Test that uncompared images form single-node chains acting as both top and bottom."""
    logger.debug("Starting test_uncompared_nodes_are_isolated_top_bottom...")
    cm = ChainManager()

    comparisons = [
        {"filename_a": "a", "filename_b": "b", "winner": "a"},
    ]
    all_filenames = {"a", "b", "isolated_1", "isolated_2"}

    cm.build(comparisons, all_filenames=all_filenames)

    # Requirement 4: Tops and bottoms behave properly for isolated nodes
    tops = cm.get_top_nodes()
    bottoms = cm.get_bottom_nodes()

    assert "isolated_1" in tops and "isolated_1" in bottoms
    assert "isolated_2" in tops and "isolated_2" in bottoms
    assert "a" in tops and "a" not in bottoms
    assert "b" in bottoms and "b" not in tops

    # They should form their own chains of length 1
    chains = list(cm.get_chains().values())
    assert ["isolated_1"] in chains
    assert ["isolated_2"] in chains
    assert ["a", "b"] in chains


def test_top_bottom_match_database_exactly() -> None:
    """Test that computed tops/bottoms exactly match DB: tops only have wins, bottoms only have losses."""
    logger.debug("Starting test_top_bottom_match_database_exactly...")

    cg = CrystalGraph()
    cg.rebuild_from_database()
    cm = cg._chain

    # Use DB functions for only-wins / only-losses
    db_tops = set(get_images_with_only_wins())
    db_bottoms = set(get_images_with_only_losses())

    # Compare with ChainManager's computed sets
    cm_tops = set(cm.get_top_nodes())
    cm_bottoms = set(cm.get_bottom_nodes())
    logger.info(f"DB tops: {len(db_tops)}, CM tops: {len(cm_tops)}")
    logger.info(f"DB bottoms: {len(db_bottoms)}, CM bottoms: {len(cm_bottoms)}")

    assert cm_tops == db_tops, (
        f"Top nodes mismatch! "
        f"DB: {len(db_tops)} tops, CM: {len(cm_tops)} tops. "
        f"Missing from CM: {db_tops - cm_tops}, Extra in CM: {cm_tops - db_tops}"
    )
    assert cm_bottoms == db_bottoms, (
        f"Bottom nodes mismatch! "
        f"DB: {len(db_bottoms)} bottoms, CM: {len(cm_bottoms)} bottoms. "
        f"Missing from CM: {db_bottoms - cm_bottoms}, Extra in CM: {cm_bottoms - db_bottoms}"
    )
