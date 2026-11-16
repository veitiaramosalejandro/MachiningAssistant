"""Minimal client for the first-stage Machining Analysis gRPC contract."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import wave

import grpc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Integration.proto import machining_analysis_v1_pb2 as pb
from Integration.proto import machining_analysis_v1_pb2_grpc as rpc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Call ModelInfo and AnalyzeSegment")
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    wav_path = args.wav.expanduser().resolve()
    if not wav_path.is_file():
        parser.error(f"WAV not found: {wav_path}")
    with grpc.insecure_channel(
        args.endpoint,
        options=[
            ("grpc.max_receive_message_length", 4 * 1024 * 1024),
            ("grpc.max_send_message_length", 4 * 1024 * 1024),
        ],
    ) as channel:
        client = rpc.MachiningAnalysisStub(channel)
        info = client.ModelInfo(pb.ModelInfoRequest(model_id="cnn-classifier"), timeout=3)
        with wave.open(str(wav_path), "rb") as source:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth(), source.getcomptype()) != (44100, 2, 2, "NONE"):
                parser.error("WAV must be 44100 Hz, stereo, PCM16")
            payload = source.readframes(info.valid_frames)
        if len(payload) != info.valid_frames * 4:
            parser.error(f"WAV must contain at least {info.valid_frames} complete frames")
        request = pb.AnalyzeSegmentRequest(
            request_id="smoke-request",
            session_id="smoke-session",
            source_recording_id=wav_path.stem,
            model=pb.ModelSelection(model_id=info.model_id, required_model_version=info.model_version),
            input_format=info.input_format,
            frame_count=info.valid_frames,
            pcm16_le=payload,
        )
        result = client.AnalyzeSegment(request, timeout=args.timeout)
    print(f"Model: {info.model_id} ({info.model_version})")
    print(f"Prediction: {result.predicted_label} ({result.confidence:.6f}, uncalibrated)")
    print("Probabilities: " + ", ".join(f"{item.class_name}={item.probability:.6f}" for item in result.class_probabilities))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
