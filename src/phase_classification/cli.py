"""Command line interface for configuration tooling.

Usage examples::

    phase-cls validate-config configs/default.yaml
    phase-cls validate-config configs/default.yaml --check-paths
    phase-cls inspect configs/default.yaml
    phase-cls train configs/default.yaml
"""

import argparse
import json
import sys
from collections.abc import Callable

from phase_classification.config import ExperimentConfig
from phase_classification.loader import ConfigLoadError, check_paths, load_config

PROG = "phase-cls"

# Label head order returned by the vendored CholecT50 loader.
LABEL_KEYS = ("triplet", "tool", "verb", "target", "phase")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG, description="Surgical video phase classification tooling."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser(
        "validate-config", help="Validate a config file against the schema."
    )
    p_validate.add_argument("config", help="Path to a YAML config file.")
    p_validate.add_argument(
        "--check-paths",
        action="store_true",
        help="Also verify the on-disk dataset layout (if data is mounted).",
    )
    p_validate.set_defaults(func=cmd_validate)

    p_inspect = sub.add_parser(
        "inspect", help="Load a config and print the resolved values."
    )
    p_inspect.add_argument("config", help="Path to a YAML config file.")
    p_inspect.add_argument(
        "--format",
        choices=("yaml", "json", "mlflow"),
        default="yaml",
        help="Output format.",
    )
    p_inspect.set_defaults(func=cmd_inspect)

    p_dl = sub.add_parser(
        "build-dataloaders",
        help="Build dataloaders for a config and print per-split sizes and a first batch shape.",
    )
    p_dl.add_argument("config", help="Path to a YAML config file.")
    p_dl.set_defaults(func=cmd_build_dataloaders)

    p_train = sub.add_parser(
        "train",
        help="Train, validate and test a model in one run, logging to MLflow.",
    )
    p_train.add_argument("config", help="Path to a YAML config file.")
    p_train.set_defaults(func=cmd_train)

    return parser


def _load(path: str) -> ExperimentConfig:
    try:
        return load_config(path)
    except ConfigLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def cmd_validate(args: argparse.Namespace) -> None:
    cfg = _load(args.config)
    print(f"OK: {args.config} is valid.")
    if args.check_paths:
        missing = check_paths(cfg)
        if missing:
            for msg in missing:
                print(f"  missing: {msg}")
            raise SystemExit(3)


def cmd_inspect(args: argparse.Namespace) -> None:
    cfg = _load(args.config)
    if args.format == "json":
        print(json.dumps(cfg.model_dump(), indent=2, default=str))
    elif args.format == "mlflow":
        print(json.dumps(cfg.to_mlflow_params(), indent=2))
    else:
        import yaml

        print(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))


def cmd_build_dataloaders(args: argparse.Namespace) -> None:
    cfg = _load(args.config)
    from phase_classification.data import build_dataloaders
    from phase_classification.data.dataloaders import DatasetUnavailableError

    try:
        dl = build_dataloaders(cfg)
    except DatasetUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc

    from torch.utils.data import DataLoader as TorchDataLoader

    for name in ("train", "val", "test"):
        loader = getattr(dl, name)
        ds = loader.dataset
        if isinstance(
            ds, list
        ):  # reference loader stores test as a list of video datasets
            print(f"{name}: {len(ds)} video sub-datasets")
            first, *_ = ds
            images, labels = next(iter(TorchDataLoader(first, batch_size=1)))
        else:
            images, labels = next(iter(loader))
            print(f"{name}: {len(ds)} samples", end="")
            first = None
        shapes = ", ".join(
            f"{key}: {tuple(t.shape)}"
            for key, t in zip(LABEL_KEYS, labels, strict=False)
        )
        if first is None:
            print(f", batch={tuple(images.shape)}")
        else:
            print(f", first batch={tuple(images.shape)}")
        print(f"    labels {shapes}")


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _load(args.config)
    from phase_classification.run import RunError, run_training

    try:
        test_metrics = run_training(cfg)
    except RunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc

    print("Test metrics:")
    for key, value in sorted(test_metrics.items()):
        print(f"  {key}: {value:.6f}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    func: Callable[[argparse.Namespace], None] = args.func
    func(args)


if __name__ == "__main__":
    main()
