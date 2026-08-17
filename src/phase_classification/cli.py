"""Command line interface for configuration tooling.

Usage examples::

    phase-cls validate-config configs/default.yaml
    phase-cls validate-config configs/default.yaml --check-paths
    phase-cls inspect configs/default.yaml
"""

import argparse
import json
import sys
from collections.abc import Callable

from phase_classification.config import ExperimentConfig
from phase_classification.loader import ConfigLoadError, check_paths, load_config

PROG = "phase-cls"


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


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    func: Callable[[argparse.Namespace], None] = args.func
    func(args)


if __name__ == "__main__":
    main()
