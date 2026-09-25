"""
tests/test_checkpoint_tracker.py
Tests for CIRCLE Checkpoint & Overfitting Tracker
"""

import os
import csv
import json
import shutil
import tempfile
import pytest
from trainer.checkpoint_tracker import CheckpointTracker


@pytest.fixture
def temp_checkpoint_dir():
    target = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trainer", "checkpoints", "test_run_tmp")
    if os.path.exists(target):
        shutil.rmtree(target, ignore_errors=True)
    os.makedirs(target, exist_ok=True)
    yield target
    shutil.rmtree(target, ignore_errors=True)


def test_csv_logging_and_columns(temp_checkpoint_dir):
    tracker = CheckpointTracker(
        checkpoint_dir=temp_checkpoint_dir,
        save_every_n_steps=100,
        csv_filename="loss_log.csv",
    )

    res1 = tracker.record_step(loss=8.2)
    res2 = tracker.record_step(loss=7.5)

    assert res1["step"] == 1
    assert res2["step"] == 2
    assert os.path.exists(tracker.csv_path)

    with open(tracker.csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert reader[0] == ["step", "loss"]
        assert reader[1] == ["1", "8.2"]
        assert reader[2] == ["2", "7.5"]


def test_best_loss_replacement(temp_checkpoint_dir):
    """
    Test scenario requested by user:
    - Step 150: loss = 7.5 (new lowest) -> saved as checkpoint_best
    - Step 151: loss = 7.9 (higher)     -> NOT saved as best
    - Step 152: loss = 6.9 (new lowest) -> saved as checkpoint_best, replacing step 150
    """
    tracker = CheckpointTracker(
        checkpoint_dir=temp_checkpoint_dir,
        save_every_n_steps=100,
    )

    # Step 150: Loss = 7.5
    res150 = tracker.record_step(loss=7.5, custom_step=150)
    assert res150["saved_best"] is True
    assert res150["best_loss"] == 7.5
    assert res150["best_step"] == 150
    assert os.path.exists(tracker.best_checkpoint_dir)

    with open(os.path.join(tracker.best_checkpoint_dir, "checkpoint_info.json")) as f:
        info150 = json.load(f)
        assert info150["step"] == 150
        assert info150["loss"] == 7.5

    # Step 151: Loss = 7.9 (higher, should NOT update best checkpoint)
    res151 = tracker.record_step(loss=7.9, custom_step=151)
    assert res151["saved_best"] is False
    assert res151["best_loss"] == 7.5
    assert res151["best_step"] == 150

    with open(os.path.join(tracker.best_checkpoint_dir, "checkpoint_info.json")) as f:
        info151 = json.load(f)
        assert info151["step"] == 150  # Still step 150!

    # Step 152: Loss = 6.9 (lower, SHOULD replace best checkpoint)
    res152 = tracker.record_step(loss=6.9, custom_step=152)
    assert res152["saved_best"] is True
    assert res152["best_loss"] == 6.9
    assert res152["best_step"] == 152

    with open(os.path.join(tracker.best_checkpoint_dir, "checkpoint_info.json")) as f:
        info152 = json.load(f)
        assert info152["step"] == 152
        assert info152["loss"] == 6.9


def test_periodic_100th_step_checkpoint(temp_checkpoint_dir):
    tracker = CheckpointTracker(
        checkpoint_dir=temp_checkpoint_dir,
        save_every_n_steps=100,
    )

    # Record 99 steps
    for step in range(1, 100):
        res = tracker.record_step(loss=10.0 - (step * 0.01))
        assert res["saved_periodic"] is False

    # Step 100
    res100 = tracker.record_step(loss=5.0)
    assert res100["saved_periodic"] is True
    assert os.path.exists(os.path.join(temp_checkpoint_dir, "checkpoint_step_100"))
