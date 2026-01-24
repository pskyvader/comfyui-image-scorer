
import pytest
import random
from training.run import around
import training.config_utils

# Mock grid_base for consistent testing
# We'll use monkeypatch if needed, or just rely on the structure if we can temporarily modify it.
# Since grid_base is a global dict, we can patch it.

@pytest.fixture
def mock_grid_base(monkeypatch):
    test_grid = {
        "test_float": {
            "type": "float",
            "min": 0.0,
            "max": 10.0,
            "step": 0.1,  # 10% change, 10% random chance
            "random": 0.5,
        },
        "test_int": {
            "type": "int",
            "min": 0,
            "max": 10,
            "step": 0.5, # 50% change
            "random": 0.5,
        },
        "test_boundary": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.2,
            "random": 0.5,
        }
    }
    # Patch the grid_base in the module where 'around' is defined and running
    monkeypatch.setattr(training.config_utils, "grid_base", test_grid)
    return test_grid


def test_around_length_and_uniqueness(mock_grid_base):
    # Test multiple times to cover random paths
    for _ in range(50):
        val = 5.0
        res = around("test_float", val)
        
        # Check length
        assert 1 <= len(res) <= 3, f"Length {len(res)} out of bounds (1-3). Res: {res}"
        
        # Check uniqueness
        assert len(res) == len(set(res)), f"Duplicates found in {res}"

def test_around_limits(mock_grid_base):
    # Test values are within min/max
    for _ in range(50):
        val = 5.0
        res = around("test_float", val)
        for r in res:
            assert 0.0 <= r <= 10.0, f"Value {r} out of bounds"

def test_around_clamping(mock_grid_base):
    # Test boundary conditions
    # Min boundary
    res = around("test_float", 0.0)
    # Should contain neighbors, but limited by min.
    # val=0. lower=0. higher=0*(1.1)=0. 
    # If random doesn't trigger, expected [0.0]. 
    # Only if random triggers we get something else.
    # To properly test neighbors:
    # Use config where step doesn't make higher 0 if val is 0?
    # val * (1+step) -> 0 * 1.1 = 0.
    
    # Let's test max boundary
    res_max = around("test_float", 10.0)
    for r in res_max:
        assert r <= 10.0

def test_around_int_type(mock_grid_base):
    for _ in range(20):
        res = around("test_int", 5)
        for r in res:
            assert isinstance(r, int) or float(r).is_integer()
            assert 0 <= r <= 10

def test_base_value_clamping(mock_grid_base):
    # If we pass a value outside limits, it should be clamped or excluded?
    # Requirement: "check that all values are within limits (including the base one)"
    # If we pass 12.0 (max 10.0), it should probably return 10.0 or values <= 10.0.
    res = around("test_float", 12.0)
    for r in res:
        assert r <= 10.0
        
    res_neg = around("test_float", -5.0)
    for r in res_neg:
        assert r >= 0.0

