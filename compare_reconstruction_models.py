"""Compare completed CNN1D-VAE and LSTM-VAE reconstruction runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    report_path = path / "test_report.json" if path.is_dir() else path
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing test_report.json: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8-sig"))


def _row(path: Path) -> dict:
    report = _read(path)
    config_path = path / "training_config.json" if path.is_dir() else path.parent / "training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.is_file() else {}
    best = report.get("best_checkpoint_evaluation", {})
    stability = report.get("stability", {})
    return {
        "model_family": report.get("model_family"),
        "run_dir": str(path if path.is_dir() else path.parent),
        "dataset_version": report.get("dataset_version"),
        "configuration_hash": report.get("configuration_hash"),
        "code_version": report.get("code_version"),
        "sequence_length": report.get("model_config", {}).get("sequence_length"),
        "latent_dim": report.get("model_config", {}).get("latent_dim"),
        "parameter_count": report.get("parameter_count"),
        "device": report.get("device"),
        "best_epoch": report.get("best_epoch"),
        "best_validation_mse": best.get("val", {}).get("reconstruction_loss"),
        "best_test_mse": best.get("test", {}).get("reconstruction_loss"),
        "final_test_mse": report.get("after", {}).get("test", {}).get("reconstruction_loss"),
        "training_seconds": report.get("training_seconds"),
        "stable": stability.get("stable"),
        "peak_cuda_memory_allocated_mb": stability.get("peak_cuda_memory_allocated_mb"),
        "best_checkpoint_selection_verified": report.get("best_checkpoint_selection_verified", False),
        "early_stopping": report.get("early_stopping", {}),
    }


def compare(cnn_run: str | Path, lstm_run: str | Path) -> dict:
    rows = [_row(Path(cnn_run).expanduser().resolve()), _row(Path(lstm_run).expanduser().resolve())]
    issues = []
    if rows[0]["dataset_version"] != rows[1]["dataset_version"]:
        issues.append("CNN and LSTM runs use different dataset versions")
    if rows[0]["sequence_length"] != rows[1]["sequence_length"]:
        issues.append("CNN and LSTM runs use different sequence lengths")
    if any(not row["best_checkpoint_selection_verified"] for row in rows):
        issues.append("At least one run has unverified best-checkpoint selection")
    if any(row["stable"] is False for row in rows):
        issues.append("At least one run contains unstable or non-finite training metrics")
    winner = None
    if not issues and all(row["best_validation_mse"] is not None for row in rows):
        winner = min(rows, key=lambda row: float(row["best_validation_mse"]))["model_family"]
    return {
        "schema_version": 1,
        "compatible": not issues,
        "issues": issues,
        "selection_metric": "best checkpoint validation reconstruction MSE",
        "winner": winner,
        "models": rows,
        "limitations": [
            "Lower reconstruction MSE does not prove better industrial fault detection.",
            "The comparison is valid only when the dataset and sequence length are identical.",
            "A supervised fault-labeled benchmark is still required for production selection.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn", type=Path, required=True, help="CNN1D-VAE run directory or test_report.json")
    parser.add_argument("--lstm", type=Path, required=True, help="LSTM-VAE run directory or test_report.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare(args.cnn, args.lstm)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
