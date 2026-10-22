"""Interactive Pygame policy playback, recorded episodes, and training metrics.

Pygame is imported only inside ``play`` so training and evaluation remain headless.
The viewer runs policy inference: watching an untrained or trained policy does not
update its weights. A training run can be monitored separately with ``run_dir``.
"""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKGROUND = (12, 17, 28)
PANEL = (21, 29, 43)
BORDER = (39, 52, 69)
TEXT = (232, 240, 250)
MUTED = (141, 157, 177)
GREEN = (80, 231, 160)
BLUE = (104, 177, 255)
RED = (255, 105, 123)
GOLD = (253, 201, 109)
ACTION_NAMES = ("LEFT", "STRAIGHT", "RIGHT")


def _state_info(state: dict[str, Any]) -> dict[str, Any]:
    """Accept both flat environment snapshots and snapshots with nested info."""
    return {**state, **state.get("info", {})}


def _snapshot(env: Any) -> dict[str, Any]:
    """Keep immutable body coordinates shared across recorded viewer frames.

    ``SnakeEnv.state`` remains the public JSON snapshot interface. Tuples also
    serialize as JSON arrays, but reusing them here avoids creating a coordinate
    list for every body cell on every step of a long episode.
    """
    state = env.state()
    state["snake"] = env.snake.copy()
    return state


def _make_policy(name: str, checkpoint: str | Path | None, seed: int) -> Any:
    from snakerl.baselines import GreedyPolicy, HamiltonianPolicy, RandomPolicy

    if name == "random":
        return RandomPolicy(seed)
    if name == "greedy":
        return GreedyPolicy()
    if name == "hamiltonian":
        return HamiltonianPolicy()
    if checkpoint is None:
        raise ValueError(f"Policy {name!r} needs a checkpoint. Supply --checkpoint.")
    from snakerl.model import NeuralPolicy

    return NeuralPolicy(checkpoint)


def _read_replay(path: Path) -> dict[str, Any]:
    replay = json.loads(path.read_text(encoding="utf-8"))
    if replay.get("version", 1) != 1:
        raise ValueError("Unsupported replay format; expected version 1.")
    if not replay.get("frames"):
        raise ValueError("The replay contains no frames.")
    if any(not isinstance(frame.get("state"), dict) for frame in replay["frames"]):
        raise ValueError("Every replay frame must contain an environment state.")
    return replay


class _RunMetrics:
    """Read a changing CSV without requiring TensorBoard or blocking playback."""

    def __init__(self, run_dir: str | Path | None):
        self.path = Path(run_dir) if run_dir else None
        self.rows: list[dict[str, str]] = []
        self.progress: dict[str, Any] = {}
        self.curve: list[tuple[float, float]] = []
        self.label = "Episode return"
        self.error = ""
        self._next_poll = 0.0
        self._csv_stamp: tuple[int, int] | None = None

    def poll(self) -> None:
        now = time.monotonic()
        if self.path is None or now < self._next_poll:
            return
        self._next_poll = now + 1.0
        try:
            progress = self.path / "progress.json"
            if progress.is_file():
                # An in-progress writer may briefly leave partial JSON. Keep the
                # most recent good snapshot until the next poll.
                self.progress = json.loads(progress.read_text(encoding="utf-8"))
            metrics = self.path / "metrics.csv"
            if not metrics.is_file():
                return
            stat = metrics.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp == self._csv_stamp:
                return
            with metrics.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.rows = rows
            self._csv_stamp = stamp
            if not rows:
                return
            normal_training = [row for row in rows if row.get("kind") == "train"
                               and row.get("curriculum_start", "0") not in ("1", "True", "true")]
            curve_rows = normal_training if any(row.get("kind") == "train" for row in rows) else rows
            fields = rows[0].keys()
            candidates = (
                "episode_return", "mean_return", "mean_reward", "ep_rew_mean",
                "rollout/ep_rew_mean", "eval/mean_reward", "reward", "return",
                "mean_coverage", "coverage", "eval/mean_coverage",
            )
            y_key = next((key for key in candidates if key in fields), None)
            if y_key is None:
                y_key = next((key for key in fields if "reward" in key or "return" in key), None)
            x_key = next((key for key in ("timesteps", "total_timesteps", "step", "steps", "episode", "update") if key in fields), None)
            curve = []
            for index, row in enumerate(curve_rows):
                try:
                    x_value = float(row[x_key]) if x_key else float(index)
                    y_value = float(row[y_key]) if y_key else float("nan")
                    if math.isfinite(x_value) and math.isfinite(y_value):
                        curve.append((x_value, y_value))
                except (TypeError, ValueError, KeyError):
                    continue
            self.curve = curve[-1000:]
            self.label = (y_key or "episode return").replace("_", " ").replace("rollout/", "").replace("eval/", "").capitalize()
            if any(row.get("kind") == "train" for row in rows):
                self.label = "Standard-start " + self.label.lower()
            self.error = ""
        except (OSError, ValueError, json.JSONDecodeError):
            self.error = "Waiting for the next metrics write"


def play(
    board_size: int = 8,
    policy_name: str = "hamiltonian",
    checkpoint: str | Path | None = None,
    seed: int = 0,
    fps: int = 12,
    replay_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    *,
    smoke_frames: int | None = None,
) -> None:
    """Watch a policy or a recorded episode with pause, step, and speed controls.

    Keys: Space pause, Right step, Left rewind (replays), Up/Down speed,
    R restart, N next seed (live policies), S save episode, Escape quit.
    ``smoke_frames`` exits after the given rendered frames for headless checks.
    """
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("Pygame is missing. Install project dependencies with uv sync.") from exc

    from snakerl.env import SnakeEnv

    replay = _read_replay(Path(replay_path)) if replay_path else None
    if replay:
        board_size = int(replay["board_size"])
        seed = int(replay.get("seed", seed))
        policy_name = str(replay.get("policy", policy_name))
    policy_name = policy_name.lower()
    baseline = policy_name in {"random", "greedy", "hamiltonian"}
    names = {"random": "Random", "greedy": "Greedy search", "hamiltonian": "Hamiltonian cycle", "ppo": "PPO from scratch", "imitation": "Imitation + PPO", "imitation_ppo": "Imitation + PPO", "imitation_only": "Expert imitation", "bc-ppo": "Imitation + PPO", "neural": "Neural policy"}
    policy_label = names.get(policy_name, policy_name.replace("_", " ").replace("-", " ").title())
    metrics = _RunMetrics(run_dir)
    metrics.poll()
    if policy_name == "neural" and metrics.progress.get("mode") in names:
        policy_label = names[metrics.progress["mode"]]

    def published_checkpoint() -> Path | None:
        value = metrics.progress.get("checkpoint")
        if not value or metrics.path is None:
            return None
        path = Path(value)
        if not path.is_absolute() and not path.is_file():
            path = metrics.path / path
        return path if path.is_file() else None

    if checkpoint is None and not baseline and not replay:
        checkpoint = published_checkpoint()
    policy = None if replay else _make_policy(policy_name, checkpoint, seed)
    loaded_checkpoint = Path(checkpoint) if checkpoint else None
    loaded_stamp = loaded_checkpoint.stat().st_mtime_ns if loaded_checkpoint and loaded_checkpoint.is_file() else None
    env = None if replay else SnakeEnv(board_size=board_size, observation_version=getattr(policy, "observation_version", 1))
    if env is not None:
        env.reset(seed=seed)
    frames = replay["frames"] if replay else [{"state": _snapshot(env), "action": None, "probabilities": None, "reward": None}]
    frame_index = 0
    total_return = 0.0
    paused = False
    done = False
    speed = max(1, min(120, int(fps)))
    probabilities: list[float] | None = None
    toast = ""
    toast_until = 0.0
    pygame.init()
    pygame.display.set_caption("SnakeRL - From zero to a full world")
    screen = pygame.display.set_mode((1180, 800), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    fonts = {
        "title": pygame.font.SysFont("segoeui", 32, bold=True),
        "big": pygame.font.SysFont("segoeui", 31, bold=True),
        "medium": pygame.font.SysFont("segoeui", 19, bold=True),
        "body": pygame.font.SysFont("segoeui", 16),
        "small": pygame.font.SysFont("segoeui", 13),
        "tiny": pygame.font.SysFont("segoeui", 11),
    }
    buttons: list[tuple[Any, str]] = []
    slider_rect = pygame.Rect(0, 0, 0, 0)
    last_step = time.monotonic()
    rendered = 0
    running = True

    def text(message: Any, x: float, y: float, color: tuple[int, int, int] = TEXT, font: str = "body", max_width: int | None = None) -> None:
        value = str(message)
        if max_width is not None:
            while value and fonts[font].size(value)[0] > max_width:
                value = value[:-2] + "…" if not value.endswith("…") else value[:-2] + "…"
        screen.blit(fonts[font].render(value, True, color), (int(x), int(y)))

    def card(rect: Any) -> None:
        pygame.draw.rect(screen, PANEL, rect, border_radius=14)
        pygame.draw.rect(screen, BORDER, rect, width=1, border_radius=14)

    def advance() -> None:
        nonlocal frame_index, done, paused, total_return, probabilities
        if replay:
            if frame_index < len(frames) - 1:
                frame_index += 1
                probabilities = frames[frame_index].get("probabilities")
            if frame_index == len(frames) - 1:
                paused = True
            return
        if done:
            return
        if hasattr(policy, "probabilities"):
            probabilities = [float(value) for value in policy.probabilities(env)]
        action = int(max(range(3), key=lambda index: probabilities[index])) if probabilities is not None else int(policy.act(env))
        _, reward, terminated, truncated, _ = env.step(action)
        total_return += float(reward)
        frames.append({"state": _snapshot(env), "action": action, "probabilities": probabilities, "reward": float(reward)})
        frame_index = len(frames) - 1
        done = bool(terminated or truncated)
        if done:
            paused = True

    def restart(next_seed: bool = False) -> None:
        nonlocal frames, frame_index, total_return, done, paused, seed, policy, probabilities, last_step
        nonlocal checkpoint, loaded_checkpoint, loaded_stamp, toast, toast_until
        if replay:
            frame_index = 0
        else:
            if next_seed:
                seed += 1
            if baseline:
                policy = _make_policy(policy_name, checkpoint, seed)
            else:
                metrics.poll()
                latest = published_checkpoint() if metrics.progress.get("status") not in {"completed", "failed"} else None
                if latest is not None:
                    stamp = latest.stat().st_mtime_ns
                    if latest != loaded_checkpoint or stamp != loaded_stamp:
                        try:
                            new_policy = _make_policy(policy_name, latest, seed)
                            if (new_policy.width, new_policy.height) != (env.width, env.height):
                                raise ValueError("The published checkpoint uses a different board size")
                            policy = new_policy
                            checkpoint = latest
                            loaded_checkpoint = latest
                            loaded_stamp = stamp
                            toast = f"Loaded newer checkpoint: {latest.name}"
                        except (OSError, RuntimeError, ValueError):
                            toast = "Checkpoint is being written; kept the current model"
                        toast_until = time.monotonic() + 6
            env.reset(seed=seed)
            frames = [{"state": _snapshot(env), "action": None, "probabilities": None, "reward": None}]
        total_return = 0.0
        frame_index = 0
        probabilities = None
        done = False
        paused = False
        last_step = time.monotonic()

    def save() -> None:
        nonlocal toast, toast_until
        folder = Path(run_dir) / "replays" if run_dir else Path("runs") / "replays"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            safe_policy = "".join(char if char.isalnum() or char in "_-" else "_" for char in policy_name)[:64] or "replay"
            path = folder / f"{safe_policy}_{board_size}x{board_size}_seed{seed}_{stamp}.json"
            result = dict(replay.get("result", {})) if replay else _state_info(frames[-1]["state"])
            if not replay:
                result["total_return"] = total_return
            replay_checkpoint = replay.get("checkpoint") if replay else None
            payload = {"version": 1, "policy": policy_name, "checkpoint": replay_checkpoint or (str(loaded_checkpoint) if loaded_checkpoint else None), "board_size": board_size, "seed": seed, "result": result, "frames": frames}
            with path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
            toast = f"Saved: {path.name}"
        except OSError as exc:
            toast = f"Could not save: {exc}"
        toast_until = time.monotonic() + 6

    def handle(action: str) -> None:
        nonlocal paused, speed, last_step, frame_index, probabilities
        if action == "pause":
            paused = not paused
            last_step = time.monotonic()
        elif action == "step":
            paused = True
            advance()
        elif action == "back" and replay:
            paused = True
            frame_index = max(0, frame_index - 1)
            probabilities = frames[frame_index].get("probabilities")
        elif action == "restart":
            restart()
        elif action == "seed" and not replay:
            restart(next_seed=True)
        elif action == "save":
            save()
        elif action == "faster":
            speed = min(120, speed + (1 if speed < 10 else 5))
        elif action == "slower":
            speed = max(1, speed - (1 if speed <= 10 else 5))

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((max(960, event.w), max(760, event.h)), pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    else:
                        action = {pygame.K_SPACE: "pause", pygame.K_RIGHT: "step", pygame.K_LEFT: "back", pygame.K_r: "restart", pygame.K_n: "seed", pygame.K_s: "save", pygame.K_UP: "faster", pygame.K_EQUALS: "faster", pygame.K_PLUS: "faster", pygame.K_KP_PLUS: "faster", pygame.K_DOWN: "slower", pygame.K_MINUS: "slower", pygame.K_KP_MINUS: "slower"}.get(event.key)
                        if action:
                            handle(action)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if replay and slider_rect.collidepoint(event.pos):
                        paused = True
                        fraction = (event.pos[0] - slider_rect.left) / max(1, slider_rect.width)
                        frame_index = min(len(frames) - 1, max(0, round(fraction * (len(frames) - 1))))
                        probabilities = frames[frame_index].get("probabilities")
                    else:
                        for rect, action in buttons:
                            if rect.collidepoint(event.pos):
                                handle(action)
                                break
            now = time.monotonic()
            if not paused and now - last_step >= 1.0 / speed:
                advance()
                last_step = now
            metrics.poll()
            current = frames[frame_index]
            state = current["state"]
            info = _state_info(state)
            snake = state.get("snake", [])
            coverage = float(info.get("coverage", len(snake) / (board_size * board_size)))
            success = bool(info.get("success", False))
            reason = info.get("termination_reason") or ""
            steps = int(info.get("steps", frame_index))
            foods = int(info.get("foods_eaten", max(0, len(snake) - 3)))
            width, height = screen.get_size()
            side_x = width - 352
            side_width = 320
            board_height = height - (320 if replay else 284)
            board_available = min(side_x - 64, board_height)
            cell_size = max(8, int(board_available / board_size))
            board_pixels = cell_size * board_size
            board_x = 34 + (side_x - 66 - board_pixels) // 2
            board_y = 152 + max(0, (board_height - board_pixels) // 2)
            screen.fill(BACKGROUND)
            text("SnakeRL", 32, 23, font="title")
            text("FROM ZERO TO A FULL WORLD", 178, 40, MUTED, "small")
            pygame.draw.line(screen, BORDER, (32, 85), (width - 32, 85))
            mode = "RECORDED EPISODE" if replay else "POLICY PLAYBACK"
            text(mode, 34, 106, BLUE, "small")
            text(f"{board_size} x {board_size} board   /   Seed {seed}", 34, 126, MUTED, "small")
            status = "FULL BOARD" if success else (str(reason).replace("_", " ").upper() if reason else ("PAUSED" if paused else "RUNNING"))
            status_color = GREEN if success else (RED if reason else (GOLD if paused else BLUE))
            status_width = fonts["small"].size(status)[0]
            text(status, side_x - status_width - 30, 108, status_color, "small")
            board_rect = pygame.Rect(board_x - 8, board_y - 8, board_pixels + 16, board_pixels + 16)
            card(board_rect)
            pygame.draw.rect(screen, (15, 23, 35), (board_x, board_y, board_pixels, board_pixels), border_radius=5)
            for row in range(board_size + 1):
                y_pos = board_y + row * cell_size
                pygame.draw.line(screen, (27, 39, 54), (board_x, y_pos), (board_x + board_pixels, y_pos))
                x_pos = board_x + row * cell_size
                pygame.draw.line(screen, (27, 39, 54), (x_pos, board_y), (x_pos, board_y + board_pixels))
            food = state.get("food")
            if food is not None:
                center = (int(board_x + (food[0] + 0.5) * cell_size), int(board_y + (food[1] + 0.5) * cell_size))
                radius = max(3, int(cell_size * 0.26))
                pygame.draw.circle(screen, (72, 35, 46), center, max(radius + 2, int(radius * 1.5)))
                pygame.draw.circle(screen, RED, center, radius)
                pygame.draw.circle(screen, (255, 187, 195), (center[0] - radius // 3, center[1] - radius // 3), max(1, radius // 4))
                pygame.draw.line(screen, GREEN, (center[0], center[1] - radius), (center[0] + max(2, radius // 3), center[1] - radius - max(2, radius // 3)), max(1, cell_size // 24))
            margin = max(1, round(cell_size * 0.07))
            for index in range(len(snake) - 1, -1, -1):
                x_coord, y_coord = snake[index]
                fade = 1.0 - index / max(1, len(snake))
                color = GREEN if index == 0 else (int(33 + 32 * fade), int(114 + 71 * fade), int(100 + 38 * fade))
                rect = pygame.Rect(board_x + x_coord * cell_size + margin, board_y + y_coord * cell_size + margin, cell_size - margin * 2, cell_size - margin * 2)
                pygame.draw.rect(screen, color, rect, border_radius=max(2, cell_size // 6))
            if snake and cell_size >= 15:
                head_x = board_x + (snake[0][0] + 0.5) * cell_size
                head_y = board_y + (snake[0][1] + 0.5) * cell_size
                direction = int(state.get("direction", 1)) % 4
                dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[direction]
                for sign in (-1, 1):
                    eye_x = head_x + dx * cell_size * 0.19 - dy * sign * cell_size * 0.17
                    eye_y = head_y + dy * cell_size * 0.19 + dx * sign * cell_size * 0.17
                    pygame.draw.circle(screen, (12, 38, 33), (round(eye_x), round(eye_y)), max(2, cell_size // 19))

            card(pygame.Rect(side_x, 106, side_width, 102))
            text("SELECTED AGENT", side_x + 18, 120, MUTED, "small")
            text(policy_label, side_x + 18, 143, font="medium", max_width=side_width - 36)
            kind = "Traditional algorithm baseline" if baseline else "Neural inference - fixed episode weights"
            text(kind, side_x + 18, 171, GREEN if baseline else BLUE, "small")
            detail = "Recorded actions" if replay else ("Algorithm actions - no model training" if baseline else f"Checkpoint: {loaded_checkpoint.name if loaded_checkpoint else 'unspecified'}")
            text(detail, side_x + 18, 189, MUTED, "tiny", max_width=side_width - 36)

            card(pygame.Rect(side_x, 222, side_width, 140))
            text("BOARD COVERAGE", side_x + 18, 235, MUTED, "small")
            text(f"{coverage:.1%}", side_x + 18, 256, GREEN, "big")
            text(f"{len(snake)} / {board_size * board_size} cells", side_x + 153, 272, MUTED, "body")
            pygame.draw.rect(screen, (39, 53, 66), (side_x + 18, 302, side_width - 36, 7), border_radius=4)
            pygame.draw.rect(screen, GREEN, (side_x + 18, 302, round((side_width - 36) * max(0, min(1, coverage))), 7), border_radius=4)
            text(f"FOOD  {foods}", side_x + 18, 325, TEXT, "small")
            text(f"STEPS  {steps:,}", side_x + 153, 325, TEXT, "small")

            card(pygame.Rect(side_x, 376, side_width, 154))
            text("ACTION PROBABILITIES" if not baseline else "LAST SELECTED ACTION", side_x + 18, 389, MUTED, "small")
            shown_probabilities = probabilities if probabilities is not None else current.get("probabilities")
            if shown_probabilities is None and current.get("action") is not None:
                shown_probabilities = [float(index == int(current["action"])) for index in range(3)]
            for index, name in enumerate(ACTION_NAMES):
                row_y = 419 + index * 32
                probability = float(shown_probabilities[index]) if shown_probabilities is not None and len(shown_probabilities) == 3 else 0.0
                text(name, side_x + 18, row_y, MUTED, "small")
                bar_x = side_x + 108
                pygame.draw.rect(screen, (38, 51, 67), (bar_x, row_y + 5, 137, 9), border_radius=4)
                pygame.draw.rect(screen, BLUE, (bar_x, row_y + 5, max(0, min(137, round(137 * probability))), 9), border_radius=4)
                text(f"{probability:.0%}" if shown_probabilities is not None else "--", side_x + 257, row_y, TEXT, "small")

            monitor_height = max(118, height - 642)
            card(pygame.Rect(side_x, 544, side_width, monitor_height))
            text("TRAINING MONITOR", side_x + 18, 558, MUTED, "small")
            if metrics.path is None:
                text("Attach a run with --run-dir", side_x + 18, 587, TEXT, "body")
                text("Training runs in a separate process.", side_x + 18, 612, MUTED, "small")
                text("TensorBoard keeps the full history.", side_x + 18, 631, MUTED, "small")
            else:
                text(metrics.label, side_x + 18, 581, TEXT, "small", max_width=side_width - 40)
                curve_rect = pygame.Rect(side_x + 20, 608, side_width - 40, max(32, monitor_height - 99))
                pygame.draw.line(screen, BORDER, curve_rect.bottomleft, curve_rect.bottomright)
                if len(metrics.curve) >= 2:
                    x_values, y_values = zip(*metrics.curve)
                    x_min, x_max = min(x_values), max(x_values)
                    y_min, y_max = min(y_values), max(y_values)
                    spread = max(1e-6, y_max - y_min)
                    points = [(curve_rect.x + round((x - x_min) / max(1, x_max - x_min) * curve_rect.width), curve_rect.bottom - round((y - y_min) / spread * (curve_rect.height - 5))) for x, y in metrics.curve]
                    pygame.draw.lines(screen, BLUE, False, points, width=2)
                    text(f"Latest {y_values[-1]:.2f}", side_x + 185, 581, BLUE, "small")
                else:
                    text("Waiting for metrics.csv", side_x + 20, 617, MUTED, "small")
                progress_status = metrics.progress.get("status", metrics.progress.get("phase", ""))
                training_steps = metrics.progress.get("timesteps", metrics.progress.get("total_timesteps", metrics.progress.get("global_step")))
                footer = str(progress_status) if progress_status else metrics.path.name
                if training_steps is not None:
                    footer += f"  /  {int(training_steps):,} training steps"
                text(footer, side_x + 18, 544 + monitor_height - 23, MUTED, "tiny", max_width=side_width - 36)

            footer_y = height - 77
            if replay:
                slider_rect = pygame.Rect(34, height - 114, side_x - 66, 15)
                pygame.draw.rect(screen, BORDER, (slider_rect.x, slider_rect.centery - 2, slider_rect.width, 4), border_radius=2)
                fraction = frame_index / max(1, len(frames) - 1)
                pygame.draw.circle(screen, BLUE, (slider_rect.x + round(fraction * slider_rect.width), slider_rect.centery), 7)
                text(f"FRAME {frame_index:,} / {len(frames) - 1:,}   -   click the timeline to seek", 34, height - 137, MUTED, "tiny")
            else:
                text("Space pause  /  Right step  /  R restart  /  N next seed  /  S save", 34, height - 109, MUTED, "small")
            buttons = []
            mouse_pos = pygame.mouse.get_pos()
            specs = [("Resume" if paused else "Pause", "pause", 91), ("Step", "step", 74), ("Restart", "restart", 87), ("Next seed", "seed", 101), ("Save replay", "save", 112), ("-", "slower", 36), ("+", "faster", 36)]
            button_x = 34
            for label, action, button_width in specs:
                rect = pygame.Rect(button_x, footer_y, button_width, 38)
                disabled = action == "seed" and bool(replay)
                hovered = rect.collidepoint(mouse_pos) and not disabled
                pygame.draw.rect(screen, (36, 62, 76) if hovered else PANEL, rect, border_radius=8)
                pygame.draw.rect(screen, BLUE if hovered else BORDER, rect, width=1, border_radius=8)
                text(label, rect.x + (rect.width - fonts["small"].size(label)[0]) // 2, rect.y + 10, (70, 84, 103) if disabled else TEXT, "small")
                buttons.append((rect, action))
                button_x += button_width + 7
            text(f"{speed} steps / s", button_x + 6, footer_y + 10, MUTED, "small")
            if toast and now < toast_until:
                rect = pygame.Rect(34, 61, side_x - 66, 25)
                pygame.draw.rect(screen, BACKGROUND, rect)
                text(toast, 34, 64, GREEN, "tiny", max_width=rect.width)
            pygame.display.flip()
            rendered += 1
            if smoke_frames is not None and rendered >= smoke_frames:
                running = False
            clock.tick(60)
    finally:
        if env is not None:
            env.close()
        pygame.quit()
