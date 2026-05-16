import json

import torch
from torch import nn

from src.eval.bigcodebench import compute_quality_metrics, make_output_stem
from src.eval.steering import steering_hook


class _Layer(nn.Module):
    def forward(self, x):
        return x


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer()])


def test_quality_metrics_joins_eval_results_by_task_and_generation_id(tmp_path):
    samples = tmp_path / "samples.jsonl"
    rows = [
        {"task_id": "BigCodeBench/0", "generation_id": 0, "solution": "def task_func():\n    return 1\n"},
        {"task_id": "BigCodeBench/0", "generation_id": 1, "solution": "def task_func():\n    return 2\n"},
    ]
    samples.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    eval_results = tmp_path / "eval_results.json"
    eval_results.write_text(json.dumps({
        "eval": {
            "BigCodeBench/0": [
                {"status": "fail"},
                {"status": "pass"},
            ]
        }
    }))

    df = compute_quality_metrics(samples, eval_results)

    assert df.loc[df["generation_id"] == 0, "passed"].item() is False
    assert df.loc[df["generation_id"] == 1, "passed"].item() is True


def test_quality_metrics_reconstructs_generation_id_when_missing(tmp_path):
    samples = tmp_path / "samples.jsonl"
    rows = [
        {"task_id": "BigCodeBench/0", "solution": "def task_func():\n    return 1\n"},
        {"task_id": "BigCodeBench/0", "solution": "def task_func():\n    return 2\n"},
    ]
    samples.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    df = compute_quality_metrics(samples)

    assert df["generation_id"].tolist() == [0, 1]


def test_steering_hook_is_context_managed_and_removed():
    model = _DummyModel()
    layer = model.model.layers[0]
    vector = torch.ones(4)

    assert len(layer._forward_hooks) == 0
    with steering_hook(model, layer=1, vector=vector, alpha=2.0):
        assert len(layer._forward_hooks) == 1
        hidden = layer(torch.zeros(1, 2, 4))
        assert torch.allclose(hidden[:, -1, :], torch.full((1, 4), 2.0))

    assert len(layer._forward_hooks) == 0


def test_make_output_stem_marks_full_vs_hard_and_steered():
    assert "bigcodebench-hard" in make_output_stem("m", "instruct", "hard", 0.8, 10, steered=False)
    assert "bigcodebench-instruct" in make_output_stem("m", "instruct", "full", 0.8, 10, steered=True)
    assert "steered" in make_output_stem("m", "instruct", "full", 0.8, 10, steered=True)
