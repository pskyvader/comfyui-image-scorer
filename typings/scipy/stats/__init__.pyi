from typing import Any, NamedTuple

class MannwhitneyuResult(NamedTuple):
    statistic: float
    pvalue: float

def mannwhitneyu(x: Any, y: Any, alternative: str = "two-sided", **kwargs: Any) -> MannwhitneyuResult: ...
