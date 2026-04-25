import random
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from src.data.filters import is_valid_syntax, passes_char_length, sha256
from src.data.metrics import CodeMetrics
from src.data.stackv2 import stream_python_files
from src.data.types import DataConfig, DatasetRow, DropCounts, Manifest, ShardInfo, StackRow
from src.utils.io import write_json


SCHEMA = pa.schema([
    ("id", pa.string()),
    ("source", pa.string()),
    ("sha256", pa.string()),
    ("cc", pa.int32()),
    ("mi", pa.float32()),
    ("comment_ratio", pa.float32()),
    ("h_volume", pa.float32()),
    ("h_difficulty", pa.float32()),
    ("h_effort", pa.float32()),
    ("sloc", pa.int32()),
    ("split", pa.string()),
])


def _assign_split(h: str) -> str:
    bucket = int(h[:8], 16) % 100
    if bucket < 80:
        return "train"
    elif bucket < 90:
        return "val"
    return "test"


def _flush_shard(rows: list[DatasetRow], output_dir: Path, idx: int) -> Path:
    path = output_dir / f"shard_{idx:04d}.parquet"
    records = [asdict(r) for r in rows]
    table = pa.table(
        {k: [r[k] for r in records] for k in SCHEMA.names},
        schema=SCHEMA,
    )
    pq.write_table(table, path)
    logger.info(f"Wrote shard {idx}: {len(rows)} rows → {path}")
    return path


def build_dataset(cfg: DataConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)

    drops = DropCounts()
    seen_hashes: set[str] = set()
    rows: list[DatasetRow] = []
    shards: list[ShardInfo] = []
    shard_idx = 0
    shard_bytes = 0
    total_rows = 0

    stream = stream_python_files(cfg)
    pbar = tqdm(stream, total=cfg.pull_count, desc="Pulling", unit="file")
    for i, row in enumerate(pbar):
        if i >= cfg.pull_count:
            break

        source: str = row.get("content") or row.get("source") or ""

        if not is_valid_syntax(source):
            drops.syntax += 1
            pbar.set_postfix(kept=total_rows, syntax=drops.syntax, char=drops.char_length, dedup=drops.dedup)
            continue

        if not passes_char_length(source, cfg.min_chars, cfg.max_chars):
            drops.char_length += 1
            pbar.set_postfix(kept=total_rows, syntax=drops.syntax, char=drops.char_length, dedup=drops.dedup)
            continue

        h = sha256(source)
        if h in seen_hashes:
            drops.dedup += 1
            pbar.set_postfix(kept=total_rows, syntax=drops.syntax, char=drops.char_length, dedup=drops.dedup)
            continue
        seen_hashes.add(h)

        try:
            metrics = CodeMetrics(source)
        except Exception:
            drops.metrics += 1
            pbar.set_postfix(kept=total_rows, syntax=drops.syntax, char=drops.char_length, dedup=drops.dedup)
            continue

        pbar.set_postfix(kept=total_rows, syntax=drops.syntax, char=drops.char_length, dedup=drops.dedup)

        file_id: str = str(row.get("hexsha") or row.get("id") or h[:16])

        rows.append(DatasetRow(
            id=file_id,
            source=source,
            sha256=h,
            cc=metrics.cc,
            mi=metrics.mi,
            comment_ratio=metrics.comment_ratio,
            h_volume=metrics.h_volume,
            h_difficulty=metrics.h_difficulty,
            h_effort=metrics.h_effort,
            sloc=metrics.sloc,
            split=_assign_split(h),
        ))
        shard_bytes += len(source.encode())
        total_rows += 1

        if shard_bytes >= cfg.shard_size_mb * 1024 * 1024:
            path = _flush_shard(rows, cfg.output_dir, shard_idx)
            shards.append(ShardInfo(path=str(path), row_count=len(rows)))
            shard_idx += 1
            rows = []
            shard_bytes = 0

    if rows:
        path = _flush_shard(rows, cfg.output_dir, shard_idx)
        shards.append(ShardInfo(path=str(path), row_count=len(rows)))

    manifest = Manifest(shards=shards, total_rows=total_rows)
    write_json(cfg.output_dir / "manifest.json", asdict(manifest))

    logger.info(f"Built {total_rows} rows across {len(shards)} shards")
    logger.info(f"Drop counts: {drops}")
