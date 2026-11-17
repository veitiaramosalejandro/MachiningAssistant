"""Create Qdrant points from trained CNN1D-VAE latent means."""
from __future__ import annotations

import csv
import hashlib
import sys
import uuid
from pathlib import Path
from typing import Any

from persistence import qdrant_collection, upsert_vectors


PROJECT_ROOT = Path(__file__).resolve().parent


def _cnn_runtime():
    cnn_root = str(PROJECT_ROOT / "CNN1D-VAE")
    if cnn_root not in sys.path:
        sys.path.insert(0, cnn_root)
    from runtime import VAEEngine  # type: ignore
    from data import read_wav  # type: ignore
    return VAEEngine, read_wav


def _point_id(training_id: str, recording_id: str, start_frame: int) -> str:
    value = f"{training_id}|{recording_id}|{start_frame}".encode("utf-8")
    return str(uuid.UUID(bytes=hashlib.md5(value).digest()))


def index_training(training_id: str, output_dir: Path, data_dir: Path, model: str, configuration: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Infer latent_mu for the sampled windows already emitted by the trainer."""
    if model != "cnn":
        return [], {"ok": False, "reason": "Qdrant embeddings are currently implemented for CNN1D-VAE only"}
    checkpoint = output_dir / "best.pt"
    prediction_path = output_dir / "window_predictions.csv"
    if not checkpoint.is_file() or not prediction_path.is_file():
        return [], {"ok": False, "reason": "best.pt or window_predictions.csv is missing"}
    VAEEngine, read_wav = _cnn_runtime()
    engine = VAEEngine.from_checkpoint(checkpoint, device=configuration.get("device", "auto"))
    rows: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    wav_cache: dict[str, Any] = {}
    with prediction_path.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    collection = qdrant_collection(int(configuration.get("latent_dim", engine.config.latent_dim)))
    for source in source_rows:
        recording_id = source["recording_id"]
        if recording_id not in wav_cache:
            wav_cache[recording_id] = read_wav(data_dir / recording_id / "audio_000000.wav")[1]
        pcm = wav_cache[recording_id]
        start_frame = int(source["start_frame"])
        length = int(engine.config.sequence_length)
        window = pcm[start_frame:start_frame + length]
        if len(window) != length:
            continue
        prediction = engine.predict_window(window, include_reconstruction=False)
        vector = [float(value) for value in prediction["latent_mu"][0]]
        point_id = _point_id(training_id, recording_id, start_frame)
        payload = {
            "training_id": training_id,
            "recording_id": recording_id,
            "source_recording_id": source.get("source_recording_id"),
            "split": source.get("split"),
            "start_frame": start_frame,
            "start_seconds": float(source.get("start_s") or 0),
            "end_seconds": float(source.get("end_s") or 0),
            "reconstruction_mse": float(source.get("trained_mse_standardized") or 0),
            "model": "CNN1D-VAE",
            "embedding_type": "latent_mu",
        }
        points.append({"id": point_id, "vector": vector, "payload": payload})
        rows.append({
            "recording_id": recording_id,
            "source_recording_id": source.get("source_recording_id"),
            "split": source.get("split"),
            "start_frame": start_frame,
            "start_s": float(source.get("start_s") or 0),
            "end_s": float(source.get("end_s") or 0),
            "score": float(source.get("trained_mse_standardized") or 0),
            "vector_id": point_id,
            "operation_context": {},
            "events": [],
        })
    upsert = upsert_vectors(collection, points)
    return rows, {"ok": True, "collection": collection, "indexed_windows": len(rows), "upsert": upsert}
