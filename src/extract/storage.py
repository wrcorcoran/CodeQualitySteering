from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.io import write_json


@dataclass
class MemmapStore:
    out_dir: Path
    model_name: str
    n_layers: int
    d_model: int
    n_rows: int
    pool_strategies: list[str]

    def __post_init__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._maps: dict[tuple[int, str], np.memmap] = {}
        for layer in range(1, self.n_layers + 1):
            for pool in self.pool_strategies:
                path = self._memmap_path(layer, pool)
                self._maps[(layer, pool)] = np.memmap(
                    path,
                    dtype=np.float16,  # stored as bf16-equivalent; numpy uses float16 dtype
                    mode="w+",
                    shape=(self.n_rows, self.d_model),
                )

    def write_row(self, row_idx: int, layer: int, pool: str, vec: np.ndarray) -> None:
        self._maps[(layer, pool)][row_idx] = vec

    def flush(self) -> None:
        for mm in self._maps.values():
            mm.flush()

    def _memmap_path(self, layer: int, pool: str) -> Path:
        return self.out_dir / f"layer_{layer:02d}_{pool}.memmap"

    def write_index(self, ids: list[str], splits: list[str]) -> None:
        table = pa.table({"id": ids, "split": splits})
        pq.write_table(table, self.out_dir / "index.parquet")

    def write_meta(self, meta: dict) -> None:
        write_json(self.out_dir / "meta.json", meta)
