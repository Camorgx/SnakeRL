"""Post-hoc trajectory analysis. These metrics never influence neural actions."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from snakerl.baselines import hamiltonian_cycle
from snakerl.evaluation import episode, write_json


def analyze_replay(replay: dict) -> dict:
    frames = replay["frames"]
    initial = frames[0]["state"]
    width, height = initial["width"], initial["height"]
    try:
        cycle = hamiltonian_cycle(width, height)
        canonical = {position: cycle[(index + 1) % len(cycle)] for index, position in enumerate(cycle)}
    except ValueError:
        canonical = None
    successors = defaultdict(set)
    agreement = 0
    moves = 0
    lower_bound = 0
    food = initial["food"]
    head = initial["snake"][0]
    distance = abs(head[0] - food[0]) + abs(head[1] - food[1]) if food else 0
    for previous, current in zip(frames, frames[1:]):
        before, after = previous["state"], current["state"]
        head_before, head_after = tuple(before["snake"][0]), tuple(after["snake"][0])
        if head_after != head_before:
            moves += 1
            successors[head_before].add(head_after)
            if canonical is not None:
                agreement += int(canonical[head_before] == head_after)
        if after["length"] > before["length"]:
            lower_bound += distance
            food = after["food"]
            distance = abs(head_after[0] - food[0]) + abs(head_after[1] - food[1]) if food else 0
    success = bool(replay["result"]["success"])
    branching = sum(len(destinations) > 1 for destinations in successors.values())
    return {
        "seed": replay["seed"], "success": success, "coverage": replay["result"]["coverage"],
        "steps": replay["result"]["steps"], "valid_moves": moves,
        "branching_positions": branching,
        "incompatible_with_any_fixed_position_cycle": bool(branching),
        "canonical_cycle_edge_agreement": agreement / moves if canonical is not None and moves else None,
        "consumed_food_manhattan_lower_bound": lower_bound,
        "steps_over_lower_bound": replay["result"]["steps"] / lower_bound if success and lower_bound else None,
        "note": "Branching during a successful episode rules out one fixed position-successor cycle. Low canonical agreement alone does not. The distance bound ignores body constraints and future-food uncertainty.",
    }


def analyze_policy(checkpoint: str | Path, board_size: int = 8, episodes: int = 20,
                   seed_start: int = 50000, output_dir: str | Path = "runs/trajectory", *, deterministic: bool = True) -> dict:
    if episodes < 1 or seed_start < 0:
        raise ValueError("episodes must be positive and seed_start nonnegative")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for path in [folder / "replay_success.json", folder / "replay_failure.json", *folder.glob("replay_adaptive_seed*.json")]:
        path.unlink(missing_ok=True)
    records = []
    chosen_success = False
    chosen_failure = False
    chosen_adaptive = False
    for seed in range(seed_start, seed_start + episodes):
        _, replay = episode("neural", board_size, seed, checkpoint, record=True, deterministic=deterministic)
        record = analyze_replay(replay)
        records.append(record)
        if record["success"] and not chosen_success:
            write_json(folder / "replay_success.json", replay)
            chosen_success = True
        if record["success"] and record["incompatible_with_any_fixed_position_cycle"] and not chosen_adaptive:
            write_json(folder / f"replay_adaptive_seed{seed}.json", replay)
            chosen_adaptive = True
        if not record["success"] and not chosen_failure:
            write_json(folder / "replay_failure.json", replay)
            chosen_failure = True
    report = {"checkpoint": str(checkpoint), "board_size": board_size, "seed_start": seed_start,
              "episodes": episodes, "successes": sum(record["success"] for record in records),
              "adaptive_successes": sum(record["success"] and record["incompatible_with_any_fixed_position_cycle"] for record in records),
              "action_selection": "deterministic_argmax" if deterministic else "seeded_policy_sampling", "records": records}
    write_json(folder / "trajectory.json", report)
    return report
