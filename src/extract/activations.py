import hashlib
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from src.data.constants import MAX_TOKENS
from src.data.types import DatasetRow
from src.extract.pool import pool_last, pool_mean
from src.extract.storage import MemmapStore
from src.models.loader import LoadedModel
from src.utils.io import read_json, write_json


POOL_STRATEGIES = ["last", "mean"]


def _chat_template_hash(tokenizer) -> str:
    template = getattr(tokenizer, "chat_template", "") or ""
    return hashlib.sha256(template.encode()).hexdigest()[:16]


def extract_activations(
    loaded: LoadedModel,
    rows: list[DatasetRow],
    out_dir: Path,
) -> None:
    cfg = loaded.cfg
    model = loaded.model
    tokenizer = loaded.tokenizer
    out_dir = out_dir / cfg.model_name
    progress_path = out_dir / "progress.json"

    start_row = 0
    if progress_path.exists():
        start_row = read_json(progress_path).get("last_completed_row", -1) + 1
        logger.info(f"Resuming from row {start_row}")

    store = MemmapStore(
        out_dir=out_dir,
        model_name=cfg.model_name,
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        n_rows=len(rows),
        pool_strategies=POOL_STRATEGIES,
    )

    store.write_meta({
        "model_name": cfg.model_name,
        "model_revision": "main",
        "tokenizer_name": cfg.hf_id,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "dtype": "bfloat16",
        "pool_strategies": POOL_STRATEGIES,
        "chat_template_hash": _chat_template_hash(tokenizer),
    })

    store.write_index(
        ids=[r.id for r in rows],
        splits=[r.split for r in rows],
    )

    batch_size = cfg.batch_size
    batches = list(range(start_row, len(rows), batch_size))

    for batch_num, batch_start in enumerate(tqdm(batches, desc=f"Extracting {cfg.model_name}")):
        batch_rows = rows[batch_start: batch_start + batch_size]

        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row.source}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch_rows
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)

        seq_len = inputs["input_ids"].shape[1]
        if seq_len > MAX_TOKENS:
            logger.info(f"Batch {batch_num+1}: skipping (padded length {seq_len} > {MAX_TOKENS})")
            continue

        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            pooled: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for l, h in enumerate(outputs.hidden_states[1:], start=1):
                last_vec = pool_last(h, mask).cpu().to(torch.bfloat16).view(torch.int16).numpy().view(np.float16)
                mean_vec = pool_mean(h, mask).cpu().to(torch.bfloat16).view(torch.int16).numpy().view(np.float16)
                pooled[l] = (last_vec, mean_vec)
            del outputs

        for i, row_idx in enumerate(range(batch_start, batch_start + len(batch_rows))):
            for l, (last_vec, mean_vec) in pooled.items():
                store.write_row(row_idx, l, "last", last_vec[i])
                store.write_row(row_idx, l, "mean", mean_vec[i])

        if (batch_num + 1) % 10 == 0:
            store.flush()
            write_json(progress_path, {"last_completed_row": batch_start + len(batch_rows) - 1})

        if batch_num < 10 or (batch_num + 1) % 10 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"Batch {batch_num+1}: GPU {allocated:.1f}GB alloc / {reserved:.1f}GB reserved")

        if (batch_num + 1) % 50 == 0:
            torch.cuda.empty_cache()

    store.flush()
    write_json(progress_path, {"last_completed_row": len(rows) - 1})
    logger.info(f"Extraction complete: {len(rows)} rows, {cfg.n_layers} layers")
