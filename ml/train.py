from __future__ import annotations

import argparse
import json
from pathlib import Path

from nebulax import acv, door, rail, shm
from nebulax.common import SUBSYSTEMS, ensure_dirs, read_json, resolve_data_root, seed_everything, write_json

MODULES = {"door": door, "acv": acv, "rail": rail, "shm": shm}
DIRECTORIES = {"door": "Door", "acv": "ACV", "rail": "Rail_Corrugation", "shm": "SHM"}


def parse_subsystems(value: str) -> list[str]:
    if value.lower() == "all":
        return list(SUBSYSTEMS)
    names = [name.strip().lower() for name in value.split(",") if name.strip()]
    invalid = set(names) - set(SUBSYSTEMS)
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown subsystem(s): {sorted(invalid)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NebulaX PS3 models")
    parser.add_argument("--data-root", required=True, help="Path to PS3/02_Datasets, PS3, or the repository root")
    parser.add_argument("--model-dir", default="models", help="Directory for trained artifacts and feature caches")
    parser.add_argument("--subsystem", default="all", type=parse_subsystems, help="all or comma-separated: door,acv,rail,shm")
    parser.add_argument("--quick", action="store_true", help="Use fewer trees for a faster smoke-test run")
    args = parser.parse_args()

    seed_everything()
    data_root = resolve_data_root(args.data_root)
    model_dir = Path(args.model_dir).expanduser().resolve()
    ensure_dirs(model_dir)
    summary_path = model_dir / "training_summary.json"
    reports = read_json(summary_path) if summary_path.exists() else {}
    for name in args.subsystem:
        print(f"\n=== Training {name.upper()} ===", flush=True)
        reports[name] = MODULES[name].train(data_root / DIRECTORIES[name], model_dir, quick=args.quick)
        print(json.dumps(reports[name], indent=2), flush=True)
    write_json(model_dir / "training_summary.json", reports)


if __name__ == "__main__":
    main()
