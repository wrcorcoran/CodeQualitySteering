from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel

from src.data.constants import MAX_CHARS, MAX_TOKENS, MIN_CHARS


class StackRow(TypedDict, total=False):
    content: str
    source: str
    hexsha: str
    id: str


class DataConfig(BaseModel):
    dataset_id: str
    dataset_name: str
    pull_count: int
    min_chars: int = MIN_CHARS
    max_chars: int = MAX_CHARS
    max_tokens: int = MAX_TOKENS
    shard_size_mb: int = 100
    seed: int = 42
    output_dir: Path


@dataclass
class DropCounts:
    syntax: int = 0
    char_length: int = 0
    dedup: int = 0
    metrics: int = 0


@dataclass
class DatasetRow:
    id: str
    source: str
    sha256: str
    cc: int
    mi: float
    comment_ratio: float
    h_volume: float
    h_difficulty: float
    h_effort: float
    sloc: int
    split: str


@dataclass
class ShardInfo:
    path: str
    row_count: int


@dataclass
class Manifest:
    shards: list[ShardInfo]
    total_rows: int
    schema_version: int = 1
