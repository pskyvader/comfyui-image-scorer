import pytest
from shared.config import config


def test_config_unique_object(isolate_config):
    """Every config file should be in a single object"""
    assert isolate_config is not None
    # Check it behaves like a mapping
    assert hasattr(isolate_config, "__getitem__")


def test_config_load_and_get_value(isolate_config):
    """get a random value from config file, compare with config loader"""
    # config.json should have "root"
    assert "root" in isolate_config
    assert isolate_config["root"] is not None


def test_config_nested_lookup(isolate_config):
    """Test nested config lookup works correctly"""
    # We suspect "prepare" is mapped to prepare_config.json
    # prepare_config.json should have "vector_schema"
    # isolate_config["prepare"] returns the dict loaded from prepare_config.json
    assert isolate_config["prepare"] is not None
    assert "vector_schema" in isolate_config["prepare"]


def test_config_cache(isolate_config):
    """Test that config values are cached and not reloaded from disk on each access"""
    # Access a value
    val1 = isolate_config["prepare"]
    # Access it again
    val2 = isolate_config["prepare"]
    # Should be same object in memory because of caching
    assert val1 is val2


def test_config_sub_section_access(isolate_config):
    """Test accessing a subsection of the config"""
    # This test assumes structure of config files.
    
    assert isolate_config["root"] is not None

    # Test deep access
    assert isolate_config["prepare"]["vector_schema"]["slots"]["cfg"] is not None

    # Check training config
    training_data = isolate_config["training"]
    
    # verify we can access top
    assert "top" in training_data
    section = training_data["top"]

    assert section is not None
    # Just check that we can access it
    assert hasattr(section, "__getitem__")


def test_force_reload_config(isolate_config):
    """Test that forcing a reload of config works correctly"""
    # Access a value
    val1 = isolate_config["prepare"]

    # Force reload by clearing cache
    isolate_config.clear()
    val2 = isolate_config["prepare"]  # this should trigger a reload from disk

    # Should not be the same object in memory after reload
    assert val1 is not val2


def test_config_modify_value(isolate_config):
    """Test modifying a config value"""
    # We need to pick a mutable part.
    # isolate_config["prepare"] returns a dict.
    cache_val = isolate_config["prepare"]["clip_device"]

    # Let's modify something deep
    isolate_config["prepare"]["clip_device"] = "test_device"
    isolate_config.clear()  # Force reload to ensure change is reflected

    # Access again
    assert isolate_config["prepare"]["clip_device"] == "test_device"

    isolate_config["prepare"]["clip_device"] = cache_val
    isolate_config.clear()  # Force reload to ensure change is reflected

    assert isolate_config["prepare"]["clip_device"] == cache_val


def test_config_no_default_value_strictly_enforced(isolate_config):
    """Getting a default value should NOT work anymore, as requested by user."""
    assert isolate_config.get("root") is not None
    
    # Missing key should raise KeyError (standard dict behavior)
    with pytest.raises(KeyError):
        isolate_config["non_existant_key"]
    
    # get(key, default) should raise ValueError
    with pytest.raises(ValueError, match="Providing a default value"):
        isolate_config.get("non_existant_key", "default")

    # even if key exists, providing default should raise ValueError (strict policy)
    with pytest.raises(ValueError, match="Providing a default value"):
        isolate_config.get("root", "default_root")


    sub_config = isolate_config.get("training")
    # Subconfigs must also respect this
    with pytest.raises(ValueError, match="Providing a default value"):
        sub_config.get("output_dir", "default")
    
