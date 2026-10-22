"""Exercise the actual Pygame loop without opening a visible window."""

import json

import pytest

pygame = pytest.importorskip("pygame")

from snakerl.env import SnakeEnv
from snakerl.evaluation import episode
from snakerl.viewer import BLUE, play
from snakerl import viewer


@pytest.fixture(autouse=True)
def _headless_display(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    yield
    pygame.quit()


def _key(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key)


def _queue_events(monkeypatch, *frame_events):
    pending = iter(frame_events)
    monkeypatch.setattr(pygame.event, "get", lambda: next(pending, []))


def _assert_export_transitions(replay):
    """A saved viewer episode must replay exactly in the seeded environment."""
    env = SnakeEnv(replay["board_size"])
    env.reset(seed=replay["seed"])
    assert replay["frames"][0]["state"] == env.state()
    assert replay["frames"][0]["action"] is None
    assert len(replay["frames"]) == replay["result"]["steps"] + 1
    total_return = 0.0
    for index, frame in enumerate(replay["frames"][1:], start=1):
        _, reward, terminated, truncated, info = env.step(frame["action"])
        assert frame["state"] == env.state()
        assert frame["reward"] == reward
        assert bool(terminated or truncated) is (index == len(replay["frames"]) - 1)
        total_return += reward
    assert info["success"] is True
    assert replay["result"]["coverage"] == 1
    assert replay["result"]["total_return"] == pytest.approx(total_return)
    env.close()


def test_baseline_live_headless_smoke_closes_pygame():
    play(board_size=8, policy_name="hamiltonian", fps=120, smoke_frames=3)
    assert not pygame.get_init()


def test_completed_training_monitor_keeps_explicit_best_model_on_new_seed(monkeypatch, tmp_path):
    best = tmp_path / "best.zip"
    final = tmp_path / "final.zip"
    best.touch()
    final.touch()
    (tmp_path / "progress.json").write_text(json.dumps({"status": "completed", "checkpoint": str(final)}))
    loaded = []

    class FakePolicy:
        width = height = 4

        def act(self, env):
            return 1

    def load(name, checkpoint, seed):
        loaded.append(checkpoint)
        return FakePolicy()

    monkeypatch.setattr(viewer, "_make_policy", load)
    _queue_events(monkeypatch, [_key(pygame.K_n)])
    play(board_size=4, policy_name="neural", checkpoint=best, run_dir=tmp_path, smoke_frames=2)
    assert loaded == [best]


def test_live_curve_does_not_mix_curriculum_success_with_standard_training(tmp_path):
    (tmp_path / "metrics.csv").write_text(
        "kind,timesteps,reward,curriculum_start\ntrain,100,100,1\ntrain,200,20,0\neval,200,30,0\n")
    monitor = viewer._RunMetrics(tmp_path)
    monitor.poll()
    assert monitor.curve == [(200.0, 20.0)]
    assert monitor.label == "Standard-start reward"


def test_pause_stops_autoplay_and_right_advances_exactly_once(monkeypatch, tmp_path):
    _queue_events(
        monkeypatch,
        [_key(pygame.K_SPACE)],
        [],
        [_key(pygame.K_RIGHT), _key(pygame.K_s)],
    )
    # At 120 steps/sec the intervening rendered frame would advance an
    # unpaused game. Save after Right must still contain only one transition.
    play(board_size=4, seed=3, fps=120, run_dir=tmp_path, smoke_frames=3)
    replay = json.loads(next((tmp_path / "replays").glob("*.json")).read_text(encoding="utf-8"))
    assert len(replay["frames"]) == 2
    assert replay["result"]["steps"] == 1
    env = SnakeEnv(4)
    env.reset(seed=3)
    assert replay["frames"][0]["state"] == env.state()
    env.step(replay["frames"][1]["action"])
    assert replay["frames"][1]["state"] == env.state()
    env.close()


def test_keyboard_pause_step_new_seed_and_save_complete_episode(monkeypatch, tmp_path):
    # Advance one old-seed frame, reset to the next seed, then step a whole
    # episode while paused. Extra Right presses after completion are harmless.
    events = [_key(pygame.K_SPACE), _key(pygame.K_RIGHT), _key(pygame.K_n), _key(pygame.K_SPACE)]
    events += [_key(pygame.K_RIGHT) for _ in range(256)]
    events.append(_key(pygame.K_s))
    _queue_events(monkeypatch, events)
    play(board_size=4, seed=5, fps=1, run_dir=tmp_path, smoke_frames=3)
    exports = list((tmp_path / "replays").glob("*.json"))
    assert len(exports) == 1
    replay = json.loads(exports[0].read_text(encoding="utf-8"))
    assert replay["seed"] == 6
    assert replay["version"] == 1
    assert replay["policy"] == "hamiltonian"
    assert replay["checkpoint"] is None
    _assert_export_transitions(replay)


def test_replay_timeline_click_and_left_key_seek_actual_display(monkeypatch, tmp_path):
    _, replay = episode("hamiltonian", board_size=4, seed=7, record=True)
    source = tmp_path / "replay.json"
    source.write_text(json.dumps(replay), encoding="utf-8")
    marker_centers = []
    original_circle = pygame.draw.circle

    def draw_circle(surface, color, center, radius, *args, **kwargs):
        # Observe the actual timeline marker while retaining real SDL drawing.
        # Its first center supplies the click location, so no window-size or
        # absolute layout coordinates are baked into this interaction test.
        if color == BLUE and radius == 7:
            marker_centers.append(tuple(center))
        return original_circle(surface, color, center, radius, *args, **kwargs)

    monkeypatch.setattr(pygame.draw, "circle", draw_circle)
    calls = 0

    def events():
        nonlocal calls
        calls += 1
        if calls == 2:
            x, y = marker_centers[0]
            return [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(x + 100, y))]
        if calls == 3:
            return [_key(pygame.K_LEFT)]
        return []

    monkeypatch.setattr(pygame.event, "get", events)
    play(replay_path=source, fps=1, smoke_frames=3)
    assert len(marker_centers) == 3
    assert marker_centers[1][0] > marker_centers[0][0]
    assert marker_centers[2][0] < marker_centers[1][0]
    assert marker_centers[2][1] == marker_centers[0][1]


def test_replay_policy_metadata_cannot_escape_export_folder(monkeypatch, tmp_path):
    _, replay = episode("hamiltonian", board_size=4, seed=7, record=True)
    replay["policy"] = "../../escaped/foreign"
    source = tmp_path / "source.json"
    source.write_text(json.dumps(replay), encoding="utf-8")
    _queue_events(monkeypatch, [_key(pygame.K_s)])
    play(replay_path=source, run_dir=tmp_path, fps=1, smoke_frames=2)
    exports = [path for path in tmp_path.rglob("*.json") if path != source]
    assert len(exports) == 1
    assert exports[0].parent == tmp_path / "replays"
    saved = json.loads(exports[0].read_text(encoding="utf-8"))
    assert saved["policy"] == replay["policy"]
    assert saved["frames"] == replay["frames"]


def test_escape_quits_without_waiting_for_smoke_limit(monkeypatch):
    calls = 0

    def events():
        nonlocal calls
        calls += 1
        return [_key(pygame.K_ESCAPE)]

    monkeypatch.setattr(pygame.event, "get", events)
    play(board_size=4, smoke_frames=1000)
    assert calls == 1
    assert not pygame.get_init()
