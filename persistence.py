"""PostgreSQL and Qdrant persistence for Machining Assistant.

PostgreSQL is the relational source of truth. Qdrant stores VAE latent vectors
for similarity search. Both integrations are lazy and best-effort so the
monitor can still run when either dependency or service is temporarily absent.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


PROJECT_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_project_env() -> dict[str, str]:
    values: dict[str, str] = {}
    if not PROJECT_ENV_PATH.is_file():
        return values
    for raw_line in PROJECT_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


PROJECT_ENV = _load_project_env()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, PROJECT_ENV.get(name, default))


POSTGRES_HOST = _env("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(_env("POSTGRES_PORT", "5432"))
POSTGRES_USER = _env("POSTGRES_USER", "user")
POSTGRES_PASSWORD = _env("POSTGRES_PASSWORD", "")
POSTGRES_DB = _env("POSTGRES_DB", "machining_assistent")
QDRANT_URL = _env("QDRANT_URL", "http://localhost:6333").rstrip("/")
QDRANT_API_KEY = _env("QDRANT_API_KEY", "")
QDRANT_COLLECTION_PREFIX = _env("QDRANT_COLLECTION_PREFIX", "machining_audio_windows_latent")


SCHEMA = """
CREATE TABLE IF NOT EXISTS recording_imports (
    job_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_root TEXT,
    manifest JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS recording_documents (
    recording_id TEXT PRIMARY KEY,
    import_job_id TEXT NOT NULL REFERENCES recording_imports(job_id) ON DELETE CASCADE,
    source_recording_id TEXT,
    split TEXT,
    source_folder TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    contexts JSONB NOT NULL DEFAULT '{}'::jsonb,
    events JSONB NOT NULL DEFAULT '{}'::jsonb,
    playback JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS training_runs (
    training_id TEXT PRIMARY KEY,
    import_job_id TEXT NOT NULL REFERENCES recording_imports(job_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    configuration JSONB NOT NULL DEFAULT '{}'::jsonb,
    report JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_dir TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS analysis_windows (
    id BIGSERIAL PRIMARY KEY,
    training_id TEXT NOT NULL REFERENCES training_runs(training_id) ON DELETE CASCADE,
    recording_id TEXT NOT NULL,
    source_recording_id TEXT,
    split TEXT,
    start_frame BIGINT,
    start_seconds DOUBLE PRECISION,
    end_seconds DOUBLE PRECISION,
    reconstruction_mse DOUBLE PRECISION,
    context JSONB NOT NULL DEFAULT '{}'::jsonb,
    events JSONB NOT NULL DEFAULT '[]'::jsonb,
    vector_id TEXT,
    vector_collection TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(training_id, recording_id, start_frame)
);
CREATE INDEX IF NOT EXISTS ix_analysis_windows_training_score
    ON analysis_windows(training_id, reconstruction_mse DESC);
CREATE INDEX IF NOT EXISTS ix_recording_documents_source
    ON recording_documents(source_recording_id);
"""


def qdrant_collection(latent_dim: int) -> str:
    if latent_dim not in (8, 16, 32):
        raise ValueError("latent_dim debe ser 8, 16 o 32")
    return f"{QDRANT_COLLECTION_PREFIX}_{latent_dim}"


def _postgres_connect():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Falta la dependencia psycopg[binary]") from exc
    return psycopg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
        connect_timeout=5,
    )


def ensure_postgres() -> dict[str, Any]:
    with _postgres_connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(SCHEMA)
            cursor.execute("SELECT current_database(), version()")
            database, version = cursor.fetchone()
    return {"ok": True, "database": database, "version": version}


def _qdrant_request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY
    request = Request(QDRANT_URL + path, data=data, headers=headers, method=method)
    with urlopen(request, timeout=5) as response:  # noqa: S310 - configured local endpoint
        raw = response.read()
        return json.loads(raw.decode("utf-8")) if raw else {}


def ensure_qdrant(latent_dim: int) -> dict[str, Any]:
    collection = qdrant_collection(latent_dim)
    try:
        _qdrant_request("GET", f"/collections/{collection}")
        return {"ok": True, "collection": collection, "created": False}
    except HTTPError as exc:
        if exc.code != 404:
            raise
    _qdrant_request("PUT", f"/collections/{collection}", {"vectors": {"size": latent_dim, "distance": "Cosine"}})
    return {"ok": True, "collection": collection, "created": True}


def health(latent_dim: int = 16) -> dict[str, Any]:
    result: dict[str, Any] = {"postgres": {"ok": False}, "qdrant": {"ok": False}}
    try:
        result["postgres"] = ensure_postgres()
    except Exception as exc:
        result["postgres"] = {"ok": False, "error": str(exc)}
    try:
        result["qdrant"] = ensure_qdrant(latent_dim)
    except Exception as exc:
        result["qdrant"] = {"ok": False, "error": str(exc)}
    return result


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {"value": value}
    except (OSError, json.JSONDecodeError):
        return {}


def persist_import(manifest: dict[str, Any]) -> dict[str, Any]:
    """Persist an import manifest and normalized JSON documents."""
    job_id = str(manifest["job_id"])
    data_dir = Path(manifest["data_dir"])
    source_root = Path(str(manifest.get("source_root", "")))
    with _postgres_connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(SCHEMA)
            cursor.execute(
                "INSERT INTO recording_imports(job_id, source, source_root, manifest) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT(job_id) DO UPDATE SET manifest=EXCLUDED.manifest",
                (job_id, manifest.get("source", "unknown"), manifest.get("source_root"), json.dumps(manifest)),
            )
            for item in manifest.get("recordings", []):
                recording_id = str(item["recording_id"])
                folder = data_dir / recording_id
                metadata = _json_file(folder / "metadata.json")
                source_folder = item.get("source_folder")
                source_dir = source_root / str(source_folder) if source_folder and source_root else folder
                contexts = _json_file(folder / "contexts.json") or _json_file(source_dir / "contexts.json")
                events = _json_file(folder / "events.json") or _json_file(source_dir / "events.json")
                playback = _json_file(folder / "playback.json") or _json_file(source_dir / "playback.json")
                cursor.execute(
                    "INSERT INTO recording_documents(recording_id, import_job_id, source_recording_id, split, source_folder, metadata, contexts, events, playback) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(recording_id) DO UPDATE SET import_job_id=EXCLUDED.import_job_id, split=EXCLUDED.split, metadata=EXCLUDED.metadata, contexts=EXCLUDED.contexts, events=EXCLUDED.events, playback=EXCLUDED.playback",
                    (recording_id, job_id, item.get("source_recording_id"), item.get("split"), item.get("source_folder"), json.dumps(metadata), json.dumps(contexts), json.dumps(events), json.dumps(playback)),
                )
    return {"ok": True, "job_id": job_id, "recordings": len(manifest.get("recordings", []))}


def persist_training(training_id: str, job: dict[str, Any], result: dict[str, Any], windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist training metrics and window-level VAE results."""
    configuration = job.get("configuration", {})
    latent_dim = int(configuration.get("latent_dim", 16))
    collection = qdrant_collection(latent_dim)
    with _postgres_connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(SCHEMA)
            cursor.execute(
                "INSERT INTO training_runs(training_id, import_job_id, model, status, configuration, report, output_dir, finished_at) VALUES (%s,%s,%s,%s,%s,%s,%s,now()) "
                "ON CONFLICT(training_id) DO UPDATE SET status=EXCLUDED.status, configuration=EXCLUDED.configuration, report=EXCLUDED.report, output_dir=EXCLUDED.output_dir, finished_at=EXCLUDED.finished_at",
                (training_id, job["import_job_id"], job.get("model", "unknown"), "completed", json.dumps(configuration), json.dumps(result), job.get("output_dir")),
            )
            for row in windows:
                cursor.execute(
                    "INSERT INTO analysis_windows(training_id, recording_id, source_recording_id, split, start_frame, start_seconds, end_seconds, reconstruction_mse, context, events, vector_id, vector_collection, payload) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(training_id, recording_id, start_frame) DO UPDATE SET reconstruction_mse=EXCLUDED.reconstruction_mse, context=EXCLUDED.context, events=EXCLUDED.events, vector_id=EXCLUDED.vector_id, vector_collection=EXCLUDED.vector_collection, payload=EXCLUDED.payload",
                    (training_id, row["recording_id"], row.get("source_recording_id"), row.get("split"), row.get("start_frame"), row.get("start_s"), row.get("end_s"), row.get("score"), json.dumps(row.get("operation_context", {})), json.dumps(row.get("events", [])), row.get("vector_id"), collection, json.dumps(row)),
                )
    return {"ok": True, "training_id": training_id, "windows": len(windows), "collection": collection}


def training_history(limit: int = 50) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 200))
    with _postgres_connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT training_id, import_job_id, model, status, configuration, report, created_at, finished_at "
                "FROM training_runs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            rows = cursor.fetchall()
    return [
        {
            "training_id": row[0],
            "import_job_id": row[1],
            "model": row[2],
            "status": row[3],
            "configuration": row[4] or {},
            "learning": (row[5] or {}).get("learning_statistics", {}),
            "created_at": row[6].isoformat() if row[6] else None,
            "finished_at": row[7].isoformat() if row[7] else None,
        }
        for row in rows
    ]


def upsert_vectors(collection: str, points: list[dict[str, Any]]) -> dict[str, Any]:
    if not points:
        return {"ok": True, "upserted": 0, "collection": collection}
    responses = []
    for offset in range(0, len(points), 256):
        responses.append(_qdrant_request(
            "PUT",
            f"/collections/{collection}/points?wait=true",
            {"points": points[offset:offset + 256]},
        ))
    return {"ok": True, "upserted": len(points), "collection": collection, "batches": len(responses)}
