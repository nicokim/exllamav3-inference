"""Shared fixtures for tests.

Most tests require a loaded ExLlamaV3 model. Set the MODEL_PATH env var
to point to your local model directory, or tests will be skipped.
"""

from __future__ import annotations

import os

import pytest
import torch


MODEL_PATH = os.environ.get("EXLLAMAV3_MODEL_PATH", "")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_model: test needs a loaded model")
    config.addinivalue_line("markers", "requires_cuda: test needs CUDA")


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def model_path():
    if not MODEL_PATH:
        pytest.skip("Set EXLLAMAV3_MODEL_PATH to run model tests")
    return MODEL_PATH


@pytest.fixture
def loaded_model(model_path):
    """Load ExLlamaV3 model, tokenizer, and cache."""
    from exllamav3 import Cache, Config, Model, Tokenizer

    config = Config.from_directory(model_path)
    model = Model.from_config(config)
    model.load(progressbar=False)

    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=2048)

    return model, tokenizer, cache
