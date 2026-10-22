import csv
import json
import math

import pytest

from snakerl.env import SnakeEnv
from snakerl.evaluation import compare, episode, evaluate, summarize, wilson_interval


@pytest.mark.parametrize("policy", ["random", "greedy", "hamiltonian"])
def test_evaluation_is_reproducible_and_preserves_seed_order(policy):
    first = evaluate(policy, board_size=4, episodes=3, seed_start=91)
    second = evaluate(policy, board_size=4, episodes=3, seed_start=91)
    assert first == second
    assert first["seeds"] == [91, 92, 93]
    assert [result["seed"] for result in first["episodes"]] == first["seeds"]
    assert first["summary"]["episodes"] == 3
    assert sum(first["summary"]["termination_counts"].values()) == 3
    assert first["neural_action_masking"] is False


def _assert_replay_reproduces_every_transition(replay):
    """Replay actual actions in a fresh seeded env, including the terminal step."""
    env = SnakeEnv(replay["board_size"])
    env.reset(seed=replay["seed"])
    frames = replay["frames"]
    assert frames[0]["state"] == env.state()
    assert frames[0]["action"] is None
    assert frames[0]["reward"] is None
    assert len(frames) == replay["result"]["steps"] + 1
    total_return = 0.0
    final_info = None
    for index, frame in enumerate(frames[1:], start=1):
        _, reward, terminated, truncated, info = env.step(frame["action"])
        assert frame["state"] == env.state()
        assert frame["state"]["steps"] == index
        assert frame["reward"] == reward
        assert env.action_space.contains(frame["action"])
        total_return += reward
        assert bool(terminated or truncated) is (index == len(frames) - 1)
        final_info = info
    assert final_info is not None
    for key, value in final_info.items():
        assert replay["result"][key] == value
    assert replay["result"]["total_return"] == pytest.approx(total_return)
    assert frames[0]["state"]["steps"] == 0  # Snapshots were not mutated later.
    env.close()


def test_success_replay_contains_all_transitions_and_final_full_board():
    result, replay = episode("hamiltonian", board_size=4, seed=7, record=True)
    assert replay is not None
    assert replay["result"] == result
    _assert_replay_reproduces_every_transition(replay)
    terminal_state = replay["frames"][-1]["state"]
    assert terminal_state["success"] is True
    assert terminal_state["coverage"] == 1
    assert terminal_state["length"] == 16
    assert terminal_state["food"] is None
    assert terminal_state["termination_reason"] == "full_board"
    assert result["foods_eaten"] == 13
    json.dumps(replay, allow_nan=False)


def test_failure_replay_includes_collision_action_and_terminal_state():
    result, replay = episode("random", board_size=4, seed=0, record=True)
    assert replay is not None
    assert result["success"] is False
    assert result["termination_reason"] in {"wall_collision", "body_collision"}
    _assert_replay_reproduces_every_transition(replay)
    assert replay["frames"][-1]["reward"] == -1.0
    assert replay["frames"][-1]["action"] is not None
    assert replay["frames"][-1]["state"]["termination_reason"] == result["termination_reason"]


def test_episode_without_recording_does_not_return_replay():
    result, replay = episode("hamiltonian", board_size=4, seed=7, record=False)
    assert result["success"] is True
    assert replay is None


def test_reusing_output_folder_does_not_keep_replays_from_previous_report(tmp_path):
    evaluate("hamiltonian", board_size=4, episodes=2, seed_start=0, output_dir=tmp_path)
    assert (tmp_path / "replay_success.json").exists()
    report = evaluate("random", board_size=4, episodes=2, seed_start=0, output_dir=tmp_path)
    assert report["summary"]["successes"] == 0
    assert (tmp_path / "replay_failure.json").exists()
    assert not (tmp_path / "replay_success.json").exists()


@pytest.mark.parametrize("policy,label", [("hamiltonian", "success"), ("random", "failure")])
def test_saved_evaluation_json_csv_and_replay_agree(tmp_path, policy, label):
    report = evaluate(policy, board_size=4, episodes=2, seed_start=0, output_dir=tmp_path)
    assert json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8")) == report
    with (tmp_path / "episodes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert [int(row["seed"]) for row in rows] == [0, 1]
    assert [int(row["steps"]) for row in rows] == [result["steps"] for result in report["episodes"]]
    replay = json.loads((tmp_path / f"replay_{label}.json").read_text(encoding="utf-8"))
    selected = next(result for result in report["episodes"] if result["seed"] == replay["seed"])
    assert replay["result"] == selected
    _assert_replay_reproduces_every_transition(replay)


def test_comparison_uses_identical_food_seeds_for_each_policy(tmp_path):
    comparison = compare(board_size=4, episodes=3, seed_start=101, models={}, output_dir=tmp_path)
    assert set(comparison["results"]) == {"random", "greedy", "hamiltonian"}
    assert comparison["seeds"] == [101, 102, 103]
    assert comparison["results"]["hamiltonian"]["successes"] == 3
    for policy in comparison["results"]:
        folder = tmp_path / policy
        report = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
        assert report["seeds"] == comparison["seeds"]
        assert [result["seed"] for result in report["episodes"]] == comparison["seeds"]
        assert report["summary"] == comparison["results"][policy]
        for replay_path in folder.glob("replay_*.json"):
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            env = SnakeEnv(4)
            env.reset(seed=replay["seed"])
            assert replay["frames"][0]["state"] == env.state()
    assert json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8")) == comparison
    assert "101–103" in (tmp_path / "comparison.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("successes,count", [(0, 1), (1, 1), (0, 100), (100, 100), (50, 100), (1, 1000)])
def test_wilson_interval_is_finite_and_contains_observed_rate(successes, count):
    lower, upper = wilson_interval(successes, count)
    assert math.isfinite(lower) and math.isfinite(upper)
    assert 0 <= lower <= upper <= 1
    assert lower - 1e-12 <= successes / count <= upper + 1e-12
    if successes == 0:
        assert lower == pytest.approx(0)
        assert upper > 0
    if successes == count:
        assert upper == pytest.approx(1)
        assert lower < 1


def test_wilson_interval_matches_known_value_and_reflects_more_evidence():
    lower, upper = wilson_interval(100, 100)
    assert lower == pytest.approx(0.9630065, abs=1e-6)
    assert upper == pytest.approx(1)
    assert wilson_interval(1000, 1000)[0] > lower
    lo_zero, hi_zero = wilson_interval(0, 100)
    assert lo_zero == pytest.approx(1 - upper)
    assert hi_zero == pytest.approx(1 - lower)


def test_summary_handles_no_successes_without_nan_and_rejects_empty_results():
    results = [episode("random", board_size=4, seed=seed)[0] for seed in (0, 1)]
    summary = summarize(results)
    assert summary["successes"] == 0
    assert summary["mean_success_steps"] is None
    assert summary["mean_coverage"] == pytest.approx(sum(result["coverage"] for result in results) / 2)
    json.dumps(summary, allow_nan=False)
    with pytest.raises(ValueError, match="episode"):
        summarize([])


@pytest.mark.parametrize("options", [{"episodes": 0}, {"batch_size": 0}])
def test_evaluation_rejects_invalid_episode_or_batch_count(options):
    with pytest.raises(ValueError, match="positive"):
        evaluate(board_size=4, **options)


def test_neural_evaluation_requires_checkpoint_before_model_import():
    with pytest.raises(ValueError, match="checkpoint"):
        evaluate("neural", board_size=4, episodes=1)
