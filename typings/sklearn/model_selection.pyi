from typing import Any, Tuple
import numpy as np

# Minimal stub for the subset of sklearn.model_selection used by the project.
# Keeps signatures simple and non-restrictive so Pylance stops reporting
# "partially unknown" for train_test_split.

def train_test_split(
    *arrays: Any,
    test_size: Any = ...,
    train_size: Any = ...,
    random_state: Any = ...,
    shuffle: bool = True,
    stratify: Any = ...,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
