from collections.abc import Iterator

from datasets import load_dataset

from src.data.types import DataConfig, StackRow


def stream_python_files(cfg: DataConfig) -> Iterator[StackRow]:
    ds = load_dataset(cfg.dataset_id, name=cfg.dataset_name, split="train", streaming=True)
    ds = ds.shuffle(seed=cfg.seed, buffer_size=10_000)
    yield from ds
