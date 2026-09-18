from __future__ import annotations

import argparse
from pathlib import Path

from nebulax import acv, door, rail, shm
from nebulax.common import SUBSYSTEMS, ensure_dirs, resolve_data_root
from train import DIRECTORIES, MODULES, parse_subsystems

OUTPUTS = {
    "door": "door_predictions.csv",
    "acv": "acv_predictions.csv",
    "rail": "rail_predictions.csv",
    "shm": "shm_predictions.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NebulaX PS3 prediction CSVs")
    parser.add_argument("--input", "--data-root", dest="data_root", required=True, help="Path to PS3/02_Datasets, PS3, or repository root")
    parser.add_argument("--model-dir", default="models", help="Directory containing trained .joblib artifacts")
    parser.add_argument("--output", "--output-dir", dest="output_dir", default="predictions", help="Prediction output directory")
    parser.add_argument("--subsystem", default="all", type=parse_subsystems, help="all or comma-separated: door,acv,rail,shm")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dirs(output_dir)
    for name in args.subsystem:
        artifact = model_dir / f"{name}.joblib"
        if not artifact.exists():
            raise FileNotFoundError(f"Missing trained artifact: {artifact}. Run train.py first.")
        output_path = output_dir / OUTPUTS[name]
        print(f"=== Predicting {name.upper()} -> {output_path.name} ===", flush=True)
        result = MODULES[name].predict(data_root / DIRECTORIES[name], model_dir, output_path)
        print(f"wrote {len(result)} rows", flush=True)


if __name__ == "__main__":
    main()

