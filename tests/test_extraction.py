import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.extract.pool import pool_last, pool_mean


MODEL_ID = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def tiny_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    return model, tokenizer


def test_hidden_states_tuple_shape(tiny_model):
    model, tokenizer = tiny_model
    inputs = tokenizer(["hello world"], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # n_layers + 1 (embedding layer)
    n_layers = model.config.num_hidden_layers
    assert len(outputs.hidden_states) == n_layers + 1
    # each hidden state: [B, T, D]
    B, T, D = outputs.hidden_states[0].shape
    assert B == 1
    assert D == model.config.hidden_size


def test_pool_last_returns_correct_shape(tiny_model):
    model, tokenizer = tiny_model
    inputs = tokenizer(["hello", "hello world"], return_tensors="pt", padding=True)
    mask = inputs["attention_mask"]
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    h = outputs.hidden_states[-1]
    result = pool_last(h, mask)
    assert result.shape == (2, model.config.hidden_size)


def test_pool_mean_returns_correct_shape(tiny_model):
    model, tokenizer = tiny_model
    inputs = tokenizer(["hello", "hello world"], return_tensors="pt", padding=True)
    mask = inputs["attention_mask"]
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    h = outputs.hidden_states[-1]
    result = pool_mean(h, mask)
    assert result.shape == (2, model.config.hidden_size)


def test_pool_mean_ignores_pad_tokens(tiny_model):
    model, tokenizer = tiny_model
    # construct a batch where one sequence is padded on the left
    inputs = tokenizer(["hi", "hi"], return_tensors="pt", padding=True)
    mask = inputs["attention_mask"]

    # manually zero out first token in second sequence to simulate padding
    h = torch.ones(2, 4, model.config.hidden_size)
    mask_zeroed = mask.clone()
    mask_zeroed[1, 0] = 0  # mark first token as pad

    result_full = pool_mean(h, mask)
    result_masked = pool_mean(h, mask_zeroed)

    # the second sequence's mean should differ once a token is masked
    assert not torch.allclose(result_full[1], result_masked[1])


def test_left_padding_last_token(tiny_model):
    model, tokenizer = tiny_model
    # with left-padding, last token position (-1) is always a real token
    inputs = tokenizer(["hi", "hello world"], return_tensors="pt", padding=True)
    mask = inputs["attention_mask"]
    # last column of mask should always be 1 (real token, not pad)
    assert mask[:, -1].all()
