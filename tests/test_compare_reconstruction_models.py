import json

from compare_reconstruction_models import compare


def _run(root, name, model, validation):
    path = root / name
    path.mkdir()
    (path / "training_config.json").write_text(json.dumps({"model_family": model}), encoding="utf-8")
    (path / "test_report.json").write_text(json.dumps({
        "model_family": model,
        "dataset_version": "dataset-1",
        "configuration_hash": f"config-{model}",
        "code_version": "code-1",
        "model_config": {"sequence_length": 2048, "latent_dim": 16},
        "parameter_count": 10,
        "device": "cpu",
        "best_epoch": 2,
        "best_checkpoint_selection_verified": True,
        "best_checkpoint_evaluation": {
            "val": {"reconstruction_loss": validation},
            "test": {"reconstruction_loss": validation + 0.1},
        },
        "after": {"test": {"reconstruction_loss": validation + 0.2}},
        "stability": {"stable": True},
    }), encoding="utf-8")
    return path


def test_comparison_selects_lower_validation_mse_on_same_dataset(tmp_path):
    cnn = _run(tmp_path, "cnn", "CNN1D-VAE", 0.4)
    lstm = _run(tmp_path, "lstm", "LSTM-VAE", 0.5)
    result = compare(cnn, lstm)
    assert result["compatible"] is True
    assert result["winner"] == "CNN1D-VAE"


def test_comparison_rejects_incompatible_sequence_lengths(tmp_path):
    cnn = _run(tmp_path, "cnn", "CNN1D-VAE", 0.4)
    lstm = _run(tmp_path, "lstm", "LSTM-VAE", 0.5)
    report = json.loads((lstm / "test_report.json").read_text())
    report["model_config"]["sequence_length"] = 4096
    (lstm / "test_report.json").write_text(json.dumps(report), encoding="utf-8")
    result = compare(cnn, lstm)
    assert result["compatible"] is False
    assert any("different sequence lengths" in issue for issue in result["issues"])
