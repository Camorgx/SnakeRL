"""Command-line entry points for the complete train/evaluate/watch loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _training_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", type=_positive, default=8, help="Square board side")
    parser.add_argument("--steps", type=_positive, default=131072)
    parser.add_argument("--envs", type=_positive, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=_positive, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy", type=float, default=0.01, help="PPO entropy bonus coefficient")
    parser.add_argument("--final-learning-rate", type=float)
    parser.add_argument("--final-entropy", type=float)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--target-kl", type=float)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--observation-version", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--reward-profile", choices=["legacy", "coverage"], default="legacy")
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--pure-ppo", action="store_true", help="Reject expert-derived or unknown checkpoint ancestry")
    parser.add_argument("--self-curriculum", type=float, default=0.0, metavar="PROBABILITY", help="Training-only resets from this PPO's own crowded states")
    parser.add_argument("--curriculum-min-coverage", type=float, default=0.7)
    parser.add_argument("--separate-encoders", action="store_true", help="Keep critic updates out of the actor CNN")
    parser.add_argument("--initial-actor-temperature", type=float, default=1.0, help="Soften initial logits before fresh PPO collection; no inference-time modification")
    parser.add_argument("--n-steps", type=_positive, default=128)
    parser.add_argument("--batch-size", type=_positive, default=256)
    parser.add_argument("--epochs", type=_positive, default=4)
    parser.add_argument("--bc-samples", type=_positive, default=30000)
    parser.add_argument("--bc-epochs", type=_positive, default=30)
    parser.add_argument("--eval-episodes", type=_positive, default=5)
    parser.add_argument("--eval-freq", type=_positive, default=16384)
    parser.add_argument("--checkpoint-freq", type=_positive, default=16384)
    parser.add_argument("--distance-reward", type=float, default=0.02)


def _config(args, output: Path, mode: str | None = None, init_checkpoint=None, size=None, steps=None, learning_rate=None):
    from snakerl.training import TrainConfig

    return TrainConfig(
        board_size=size or args.size, total_timesteps=steps or args.steps,
        n_envs=args.envs, seed=args.seed, output_dir=str(output),
        device=args.device, threads=args.threads, vec_env=args.vec_env,
        learning_rate=learning_rate if learning_rate is not None else args.learning_rate,
        ent_coef=args.entropy,
        final_learning_rate=args.final_learning_rate, final_ent_coef=args.final_entropy,
        gamma=args.gamma, gae_lambda=args.gae_lambda, target_kl=args.target_kl, clip_range=args.clip_range,
        observation_version=args.observation_version, reward_profile=args.reward_profile,
        random_start=args.random_start, require_pure_ppo=args.pure_ppo,
        self_curriculum_probability=args.self_curriculum, curriculum_min_coverage=args.curriculum_min_coverage,
        share_features_extractor=not args.separate_encoders,
        initial_actor_temperature=args.initial_actor_temperature,
        n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.epochs,
        mode=mode or args.mode, bc_samples=args.bc_samples, bc_epochs=args.bc_epochs,
        eval_episodes=args.eval_episodes, eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq, init_checkpoint=str(init_checkpoint) if init_checkpoint else None,
        distance_reward=args.distance_reward,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="SnakeRL - train, evaluate, compare and watch real policies")
    commands = result.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train PPO or expert imitation followed by PPO")
    _training_options(train)
    train.add_argument("--mode", choices=["ppo", "imitation_ppo"], default="ppo")
    train.add_argument("--out", type=Path, default=Path("runs/ppo8"))
    train.add_argument("--init-checkpoint", "--resume", dest="init_checkpoint", type=Path,
                       help="Initialize weights from a checkpoint; optimizer and timestep counter restart")
    curriculum = commands.add_parser("curriculum", help="Transfer CNN weights through progressively larger boards")
    _training_options(curriculum)
    curriculum.add_argument("--sizes", nargs="+", type=_positive, default=[4, 6, 8])
    curriculum.add_argument("--mode", choices=["ppo", "imitation_ppo"], default="ppo")
    curriculum.add_argument("--out", type=Path, default=Path("runs/curriculum"))
    experiment = commands.add_parser("experiment", help="Run scratch PPO, imitation+PPO, and matched-seed comparison")
    _training_options(experiment)
    experiment.add_argument("--finetune-steps", type=_positive, default=16384)
    experiment.add_argument("--finetune-learning-rate", type=float, default=1e-5)
    experiment.add_argument("--episodes", type=_positive, default=100)
    experiment.add_argument("--seed-start", type=int, default=10000)
    experiment.add_argument("--out", type=Path, default=Path("runs/experiment8"))
    evaluate = commands.add_parser("evaluate", help="Held-out evaluation and successful/failed episode replays")
    evaluate.add_argument("--policy", choices=["random", "greedy", "hamiltonian", "neural"], default="hamiltonian")
    evaluate.add_argument("--size", type=_positive, default=8)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--episodes", type=_positive, default=100)
    evaluate.add_argument("--seed-start", type=int, default=10000)
    evaluate.add_argument("--batch-size", type=_positive, default=32)
    evaluate.add_argument("--out", type=Path, default=Path("runs/evaluation"))
    evaluate.add_argument("--stochastic", action="store_true", help="Sample the neural actor with a reproducible independent per-episode RNG")
    compare = commands.add_parser("compare", help="Race baselines and named models on exactly the same seed set")
    compare.add_argument("--size", type=_positive, default=8)
    compare.add_argument("--episodes", type=_positive, default=100)
    compare.add_argument("--seed-start", type=int, default=10000)
    compare.add_argument("--model", action="append", default=[], metavar="NAME=CHECKPOINT")
    compare.add_argument("--out", type=Path, default=Path("runs/comparison"))
    play = commands.add_parser("play", help="Pygame live model playback or recorded failure replay")
    play.add_argument("--policy", choices=["random", "greedy", "hamiltonian", "neural"], default="hamiltonian")
    play.add_argument("--size", type=_positive, default=8)
    play.add_argument("--checkpoint", type=Path)
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--fps", type=_positive, default=12)
    play.add_argument("--replay", type=Path)
    play.add_argument("--run-dir", type=Path)
    plot = commands.add_parser("plot", help="Export charts from actual training or comparison logs")
    source = plot.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--comparison", type=Path)
    plot.add_argument("--out", type=Path)
    analyze = commands.add_parser("analyze", help="Check successful neural trajectories for fixed-cycle behavior and detours")
    analyze.add_argument("--checkpoint", type=Path, required=True)
    analyze.add_argument("--size", type=_positive, default=8)
    analyze.add_argument("--episodes", type=_positive, default=20)
    analyze.add_argument("--seed-start", type=int, default=50000)
    analyze.add_argument("--out", type=Path, default=Path("runs/trajectory"))
    analyze.add_argument("--stochastic", action="store_true")
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        if args.command == "experiment" and args.pure_ppo:
            raise ValueError("experiment includes expert imitation; use train or curriculum with --pure-ppo")
        if args.command in {"train", "curriculum", "experiment"}:
            from snakerl.training import train

            if args.command == "train":
                print(train(_config(args, args.out, init_checkpoint=args.init_checkpoint)))
            elif args.command == "curriculum":
                checkpoint = None
                for size in args.sizes:
                    checkpoint = train(_config(args, args.out / f"{size}x{size}", init_checkpoint=checkpoint, size=size))
                print(checkpoint)
            else:
                from snakerl.evaluation import compare
                from snakerl.reporting import plot_comparison, plot_run

                scratch = train(_config(args, args.out / "ppo", mode="ppo"))
                imitation = train(_config(args, args.out / "imitation_ppo", mode="imitation_ppo",
                                          steps=args.finetune_steps, learning_rate=args.finetune_learning_rate))
                models = {"ppo": scratch, "imitation_ppo": imitation}
                bc_model = args.out / "imitation_ppo" / "checkpoints" / "bc.zip"
                if bc_model.exists():
                    models["imitation_only"] = bc_model
                comparison = compare(args.size, args.episodes, args.seed_start, models, args.out / "comparison")
                plot_comparison(args.out / "comparison" / "comparison.json")
                for run_dir in [args.out / "ppo", args.out / "imitation_ppo"]:
                    plot_run(run_dir)
                print(json.dumps(comparison["results"], indent=2))
        elif args.command == "evaluate":
            from snakerl.evaluation import evaluate

            report = evaluate(args.policy, args.size, args.episodes, args.seed_start, args.checkpoint, args.out, args.batch_size,
                              deterministic=not args.stochastic)
            print(json.dumps(report["summary"], indent=2))
        elif args.command == "compare":
            from snakerl.evaluation import compare
            from snakerl.reporting import plot_comparison

            models = {}
            for model in args.model:
                if "=" not in model:
                    raise ValueError("--model must be NAME=CHECKPOINT")
                label, path = model.split("=", 1)
                if not label or label in {"random", "greedy", "hamiltonian"} or label in models:
                    raise ValueError("Model names must be unique and distinct from baseline names.")
                if Path(label).name != label or label in {".", ".."} or ":" in label:
                    raise ValueError("Model name must be a simple directory name.")
                models[label] = path
            report = compare(args.size, args.episodes, args.seed_start, models, args.out)
            plot_comparison(args.out / "comparison.json")
            print(json.dumps(report["results"], indent=2))
        elif args.command == "play":
            from snakerl.viewer import play

            play(args.size, args.policy, args.checkpoint, args.seed, args.fps, args.replay, args.run_dir)
        elif args.command == "plot":
            from snakerl.reporting import plot_comparison, plot_run

            print(plot_run(args.run_dir, args.out) if args.run_dir else plot_comparison(args.comparison, args.out))
        elif args.command == "analyze":
            from snakerl.trajectory import analyze_policy
            report = analyze_policy(args.checkpoint, args.size, args.episodes, args.seed_start, args.out,
                                    deterministic=not args.stochastic)
            print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    except (ValueError, OSError) as exc:
        raise SystemExit(f"SnakeRL: {exc}") from exc


if __name__ == "__main__":
    main()
