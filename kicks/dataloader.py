"""DataLoader subclass for kick drum audio batches."""

from typing import Any

from torch.utils.data import DataLoader

from .dataset import KickDataset


class KickDataloader(DataLoader):
    """Thin wrapper around DataLoader for kick drum samples.

    Exists as an extension point for custom batching or iteration logic.
    """

    def __init__(self, dataset: KickDataset, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset, *args, **kwargs)

    def __iter__(self) -> Any:
        return super().__iter__()
