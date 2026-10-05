"""
Dataset handler registry for PDB (Precise Debugging Benchmarking).

NOTE: We use a simple dictionary rather than automatic module discovery
(importlib + pkgutil) because there are only a handful of datasets. An explicit registry
is easier to read, debug, and extend. When adding a new dataset, you add one import and
one dict entry — see dataset/README.md for full instructions.
"""

from dataset.bigcodebench.handler import BigCodeBenchHandler
from dataset.livecodebench.handler import LiveCodeBenchHandler
from dataset.swesmith.handler import SWESmithHandler
from dataset.base import DatasetHandler

_REGISTRY = {
    "bigcodebench": BigCodeBenchHandler(),
    "livecodebench": LiveCodeBenchHandler(),
    "swesmith": SWESmithHandler(),
}


def get_handler(dataset_name):
    """
    Look up the handler for a dataset by its canonical string name.

    NOTE: [edge case callout] Raises KeyError with a helpful message listing all known
    datasets if the name is not found.
    """
    if dataset_name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. Known datasets: {known}. "
            f"See dataset/README.md for how to add a new one."
        )
    return _REGISTRY[dataset_name]
