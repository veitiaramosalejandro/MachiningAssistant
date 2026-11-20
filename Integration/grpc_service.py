"""First-stage gRPC service for the resident classification engine.

This implementation intentionally exposes only the unary ModelInfo and
AnalyzeSegment RPCs. AnalyzeStream remains a separate phase because it needs
session lifetime, bounded buffering, reconnect semantics, and backpressure.
"""
from __future__ import annotations

import argparse
from concurrent import futures
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from typing import Final

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Integration.proto import machining_analysis_v1_pb2 as pb
from Integration.proto import machining_analysis_v1_pb2_grpc as rpc
from Integration.simulate_monitor import load_engine


SAMPLE_RATE: Final = 44100
CHANNELS: Final = 2
VALID_FRAMES: Final = 220500
FRAME_BYTES: Final = CHANNELS * 2
PCM_BYTES: Final = VALID_FRAMES * FRAME_BYTES
MAX_RPC_MESSAGE_BYTES: Final = 4 * 1024 * 1024
PROTOCOL_REVISION: Final = "1.0-draft"
ENV_FILE = PROJECT_ROOT / ".env"


def _load_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """Load simple KEY=VALUE settings without requiring python-dotenv."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env line {number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"Invalid .env key on line {number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def _model_version(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()[:16]}"


def _abort(context, status, message):
    context.abort(status, message)


def _validate_format(value: pb.AudioFormat, context) -> None:
    if value.sample_rate_hz != SAMPLE_RATE or value.channel_count != CHANNELS:
        _abort(context, grpc.StatusCode.INVALID_ARGUMENT,
               "input_format must be 44100 Hz with 2 channels")
    if value.encoding != pb.PCM_ENCODING_S16LE_INTERLEAVED:
        _abort(context, grpc.StatusCode.INVALID_ARGUMENT,
               "input_format.encoding must be S16LE_INTERLEAVED")


def _diagnostics(engine) -> pb.ModelDiagnostics:
    result = pb.ModelDiagnostics(
        confidence_calibrated=False,
        cnc_context_used_by_model=False,
        label_semantics=getattr(engine, "label_metadata", {}).get(
            "label_semantics", "unverified_classification_label"),
        diagnostic_only=True,
        warnings=[
            "Confidence is an uncalibrated softmax score.",
            "CNC context is not used as a model feature.",
            "This service does not issue machine-control commands.",
        ],
    )
    metadata = getattr(engine, "label_metadata", {})
    if "synthetic_only" in metadata:
        result.synthetic_only = bool(metadata["synthetic_only"])
    return result


class MachiningAnalysisService(rpc.MachiningAnalysisServicer):
    def __init__(self, engine, model_id: str, checkpoint: Path):
        if engine.sample_rate != SAMPLE_RATE or engine.valid_frames != VALID_FRAMES:
            raise ValueError("Checkpoint does not match the 44.1 kHz five-second contract")
        self.engine = engine
        self.model_id = model_id
        self.checkpoint = checkpoint
        self.model_version = _model_version(checkpoint)
        self.diagnostics = _diagnostics(engine)

    def _info(self) -> pb.ModelInfoResponse:
        return pb.ModelInfoResponse(
            protocol_revision=PROTOCOL_REVISION,
            model_id=self.model_id,
            model_version=self.model_version,
            model_family=self.engine.model_family,
            input_format=pb.AudioFormat(
                sample_rate_hz=SAMPLE_RATE,
                channel_count=CHANNELS,
                encoding=pb.PCM_ENCODING_S16LE_INTERLEAVED,
                channel_descriptions=[
                    pb.ChannelDescription(index=0, logical_name="sound", physical_quantity="sound", numeric_unit="pcm_counts"),
                    pb.ChannelDescription(index=1, logical_name="vibration", physical_quantity="vibration", numeric_unit="pcm_counts"),
                ],
            ),
            valid_frames=VALID_FRAMES,
            result_hop_frames=VALID_FRAMES,
            classes=[pb.ClassDefinition(index=i, name=name)
                     for i, name in enumerate(self.engine.class_names)],
            diagnostics=self.diagnostics,
            recommended_chunk_frames=882,
            max_pcm_chunk_frames=VALID_FRAMES,
            max_rpc_message_bytes=MAX_RPC_MESSAGE_BYTES,
        )

    def ModelInfo(self, request, context):
        if request.model_id and request.model_id != self.model_id:
            _abort(context, grpc.StatusCode.NOT_FOUND,
                   f"Unknown model_id: {request.model_id}")
        return self._info()

    def AnalyzeSegment(self, request, context):
        if not request.request_id or not request.session_id or not request.source_recording_id:
            _abort(context, grpc.StatusCode.INVALID_ARGUMENT,
                   "request_id, session_id and source_recording_id are required")
        if request.model.model_id != self.model_id:
            _abort(context, grpc.StatusCode.NOT_FOUND,
                   f"Unknown model_id: {request.model.model_id}")
        if request.model.required_model_version and request.model.required_model_version != self.model_version:
            _abort(context, grpc.StatusCode.FAILED_PRECONDITION,
                   "required_model_version does not match the deployed checkpoint")
        _validate_format(request.input_format, context)
        if request.frame_count != VALID_FRAMES:
            _abort(context, grpc.StatusCode.INVALID_ARGUMENT,
                   f"frame_count must equal {VALID_FRAMES}")
        if len(request.pcm16_le) != PCM_BYTES:
            _abort(context, grpc.StatusCode.INVALID_ARGUMENT,
                   f"pcm16_le must contain exactly {PCM_BYTES} bytes")

        try:
            import numpy as np
            pcm = np.frombuffer(request.pcm16_le, dtype="<i2").reshape(VALID_FRAMES, CHANNELS)
            prediction = self.engine.predict_pcm(pcm, sample_rate=SAMPLE_RATE)
        except (ValueError, RuntimeError) as exc:
            _abort(context, grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        completed = Timestamp()
        completed.FromDatetime(datetime.now(timezone.utc))
        result_id = hashlib.sha256(
            f"{request.session_id}|{self.model_version}|{request.first_source_frame}|{VALID_FRAMES}".encode()
        ).hexdigest()[:24]
        output = pb.SegmentResult(
            request_id=request.request_id,
            session_id=request.session_id,
            source_recording_id=request.source_recording_id,
            result_id=result_id,
            model_id=self.model_id,
            model_version=self.model_version,
            window_index=0,
            first_source_frame=request.first_source_frame,
            end_source_frame_exclusive=request.first_source_frame + VALID_FRAMES,
            valid_frames=VALID_FRAMES,
            predicted_class_index=prediction["predicted_class_index"],
            predicted_label=prediction["predicted_label"],
            confidence=prediction["confidence"],
            diagnostics=self.diagnostics,
            context_alignment_complete=False,
            server_completed_at_utc=completed,
        )
        output.class_probabilities.extend(
            pb.ClassProbability(class_index=i, class_name=name, probability=probability)
            for i, (name, probability) in enumerate(prediction["class_probabilities"].items())
        )
        return output


def create_server(service: MachiningAnalysisService, host: str, port: int) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        options=[
            ("grpc.max_receive_message_length", MAX_RPC_MESSAGE_BYTES),
            ("grpc.max_send_message_length", MAX_RPC_MESSAGE_BYTES),
        ],
    )
    rpc.add_MachiningAnalysisServicer_to_server(service, server)
    bound = server.add_insecure_port(f"{host}:{port}")
    if bound != port:
        raise OSError(f"Could not bind gRPC endpoint {host}:{port}")
    return server


def main(argv=None) -> int:
    env = _load_env_file()

    def env_value(name: str, fallback: str) -> str:
        return env.get(name, fallback)

    checkpoint_value = env_value("MACHINING_GRPC_CHECKPOINT", "")
    parser = argparse.ArgumentParser(description="Run the first-stage Machining Analysis gRPC service")
    parser.add_argument("--model", choices=("cnn", "lstm"), default=env_value("MACHINING_GRPC_MODEL", "cnn"))
    parser.add_argument("--checkpoint", type=Path, default=Path(checkpoint_value) if checkpoint_value else None)
    parser.add_argument("--host", default=env_value("MACHINING_GRPC_HOST", "127.0.0.1"), help="Default is local-only; use TLS before remote binding")
    parser.add_argument("--port", type=int, default=int(env_value("MACHINING_GRPC_PORT", "50051")))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=env_value("MACHINING_GRPC_DEVICE", "auto"))
    parser.add_argument("--threads", type=int, default=int(env_value("MACHINING_GRPC_THREADS", "2")))
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or args.threads < 1:
        parser.error("port must be 1..65535 and threads must be positive")
    if args.checkpoint is None or not str(args.checkpoint):
        parser.error("A checkpoint is required; set MACHINING_GRPC_CHECKPOINT in .env or pass --checkpoint")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    import torch
    torch.set_num_threads(args.threads)
    engine = load_engine(args.model, checkpoint, device=args.device)
    service = MachiningAnalysisService(engine, f"{args.model}-classifier", checkpoint)
    server = create_server(service, args.host, args.port)
    server.start()
    print(f"Machining Analysis gRPC listening on {args.host}:{args.port}", flush=True)
    print(f"Model: {service.model_id}; version: {service.model_version}; device: {engine.device}", flush=True)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
