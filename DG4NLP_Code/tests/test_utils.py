"""Phase 3 tests: configuration and shared utilities."""

import random

import numpy as np

import config
from src.utils import Timer, load_json, save_json, set_seed


def test_paths_are_inside_project():
    for d in config.ALL_DIRS:
        assert config.PROJECT_ROOT in d.parents


def test_split_proportions_are_valid():
    assert 0 < config.TEST_SIZE < 0.5
    assert 0 < config.VAL_SIZE < 0.5
    assert config.TEST_SIZE + config.VAL_SIZE < 1


def test_set_seed_makes_random_repeatable():
    set_seed(123)
    a = (random.random(), np.random.rand())
    set_seed(123)
    b = (random.random(), np.random.rand())
    assert a == b


def test_timer_measures_positive_time():
    with Timer() as t:
        sum(range(10_000))
    assert t.elapsed > 0


def test_json_roundtrip(tmp_path):
    data = {"model": "svm", "macro_f1": 0.5, "labels": ["cs.LG", "hep-ph"]}
    path = tmp_path / "sub" / "x.json"
    save_json(data, path)
    assert load_json(path) == data


def test_public_settings_contains_no_secret():
    settings = config.public_settings()
    assert "LLM_API_KEY" not in settings
    assert settings["SEED"] == config.SEED
