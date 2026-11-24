"""TLS socket server compatible with MachiningMonitor's Analyzer protocol.

MachiningMonitor opens the connection to the Analyzer.  This module exposes
the Python side as a small, reusable TLS server and deliberately keeps model
execution outside the transport layer.  Callbacks receive validated payloads
and return either a state string or an ``{"state": ...}`` dictionary.

Wire compatibility with ``MachiningMonitor.Contracts.SocketProtocol``:

* four-byte unsigned big-endian frame length;
* UTF-8 JSON envelope, protocol version 1;
* ``requestId`` correlation;
* maximum frame size of 1 MiB.
"""
from __future__ import annotations

import asyncio
import argparse
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import inspect
import json
import math
from pathlib import Path
import subprocess
import ssl
import sys
import time
from typing import Any, Final, TypeAlias
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import UUID


MAXIMUM_FRAME_BYTES: Final = 1024 * 1024
PROTOCOL_VERSION: Final = 1
ACCEPTED_MESSAGE_TYPE: Final = "analysis.accepted"
MODEL_SAMPLE_RATE_HZ: Final = 44100
MODEL_CHANNELS: Final = 2
MODEL_SAMPLE_BYTES: Final = 2
DEFAULT_AUDIO_CHUNK_MS: Final = 20
DEFAULT_FFMPEG: Final = "ffmpeg"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_CERTIFICATE: Final = PROJECT_ROOT / "certs" / "machining-assistant-analyzer.crt"
DEFAULT_PRIVATE_KEY: Final = PROJECT_ROOT / "certs" / "machining-assistant-analyzer.key"

Payload: TypeAlias = dict[str, Any]
HandlerResult: TypeAlias = str | Payload
MessageHandler: TypeAlias = Callable[[Payload], HandlerResult | Awaitable[HandlerResult]]
AudioChunkHandler: TypeAlias = Callable[[Payload], None | Awaitable[None]]


def _audio_source_url(url: str, username: str, password: str) -> str:
    """Add credentials only to the child FFmpeg command, never to logs."""
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.port:
        host += f":{parsed.port}"
    user_info = f"{quote(username, safe='')}:{quote(password, safe='')}@"
    return urlunsplit((parsed.scheme, user_info + host, parsed.path, parsed.query, ""))


def _public_audio_url(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or "<invalid-host>"
    if parsed.port:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, ""))


class _AudioStreamWorker:
    """Read one RTSP source and emit model-compatible PCM chunks."""

    def __init__(
        self,
        *,
        session_id: str,
        url: str,
        username: str,
        password: str,
        ffmpeg: str,
        transport: str,
        chunk_ms: int,
        context_provider: Callable[[], Payload],
        on_audio: AudioChunkHandler | None,
        logger: Callable[[str], None],
    ) -> None:
        if transport not in {"tcp", "udp"}:
            raise ValueError("audio transport must be tcp or udp")
        if chunk_ms <= 0:
            raise ValueError("audio chunk duration must be positive")
        self.session_id = session_id
        self.url = url
        self.username = username
        self.password = password
        self.ffmpeg = ffmpeg
        self.transport = transport
        self.chunk_frames = max(1, round(MODEL_SAMPLE_RATE_HZ * chunk_ms / 1000))
        self.chunk_bytes = self.chunk_frames * MODEL_CHANNELS * MODEL_SAMPLE_BYTES
        self.context_provider = context_provider
        self.on_audio = on_audio
        self.logger = logger
        self.ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._process: asyncio.subprocess.Process | None = None
        self.frames_emitted = 0
        self._receiving = False

    @property
    def is_receiving(self) -> bool:
        return self._receiving and not self._stop.is_set()

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("audio worker is already running")
        self._task = asyncio.create_task(self._run(), name=f"rtsp-audio-{self.session_id}")

    async def wait_ready(self, timeout: float = 3.0) -> None:
        await asyncio.wait_for(self.ready.wait(), timeout=timeout)

    async def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None and process.returncode is None:
            process.terminate()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            process: asyncio.subprocess.Process | None = None
            stderr_task: asyncio.Task[bytes] | None = None
            try:
                command = [
                    self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                    "-rtsp_transport", self.transport, "-timeout", "2000000",
                    "-probesize", "32768", "-analyzeduration", "100000",
                    "-i", _audio_source_url(self.url, self.username, self.password),
                    "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
                    "-ar", str(MODEL_SAMPLE_RATE_HZ), "-ac", str(MODEL_CHANNELS), "pipe:1",
                ]
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self._process = process
                assert process.stdout is not None
                assert process.stderr is not None
                stderr_task = asyncio.create_task(process.stderr.read())
                pending = b""
                while not self._stop.is_set():
                    chunk = await process.stdout.read(self.chunk_bytes)
                    if not chunk:
                        break
                    pending += chunk
                    complete = len(pending) - (len(pending) % (MODEL_CHANNELS * MODEL_SAMPLE_BYTES))
                    while complete >= self.chunk_bytes:
                        pcm = pending[:self.chunk_bytes]
                        pending = pending[self.chunk_bytes:]
                        first_frame = self.frames_emitted
                        self.frames_emitted += self.chunk_frames
                        self._receiving = True
                        self.ready.set()
                        if self.on_audio is not None:
                            result = self.on_audio({
                                "sessionId": self.session_id,
                                "pcm16le": pcm,
                                "sampleRateHz": MODEL_SAMPLE_RATE_HZ,
                                "channelCount": MODEL_CHANNELS,
                                "firstFrame": first_frame,
                                "positionSeconds": first_frame / MODEL_SAMPLE_RATE_HZ,
                                "context": self.context_provider(),
                            })
                            if inspect.isawaitable(result):
                                await result
                        complete = len(pending) - (len(pending) % (MODEL_CHANNELS * MODEL_SAMPLE_BYTES))
                await process.wait()
                error = (await stderr_task).decode("utf-8", errors="replace").strip() if stderr_task else ""
                if error and not self._stop.is_set():
                    self.logger(
                        f"RTSP audio session {self.session_id} ended for {_public_audio_url(self.url)}: "
                        f"{error.replace(_audio_source_url(self.url, self.username, self.password), _public_audio_url(self.url))[-500:]}"
                    )
            except asyncio.CancelledError:
                raise
            except (OSError, RuntimeError) as exc:
                if not self._stop.is_set():
                    self.logger(f"RTSP audio session {self.session_id} unavailable: {type(exc).__name__}")
            except Exception as exc:
                if not self._stop.is_set():
                    self.logger(f"RTSP audio session {self.session_id} failed: {type(exc).__name__}")
            finally:
                self._receiving = False
                self._process = None
                if process is not None and process.returncode is None:
                    process.kill()
                    await process.wait()
                if stderr_task is not None and not stderr_task.done():
                    stderr_task.cancel()
            if not self._stop.is_set():
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=0.25)
                except asyncio.TimeoutError:
                    pass


@dataclass
class _AnalysisSession:
    start: Payload
    state: str
    lease_deadline: float
    sequence: int = 0
    last_data: str | None = None
    latest_context: Payload | None = None
    latest_message: Payload | None = None
    audio: _AudioStreamWorker | None = None


def _require_object(value: Any, name: str) -> Payload:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _uuid(value: Any, field: str, *, allow_empty: bool = False) -> str:
    try:
        parsed = UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a UUID") from exc
    if not allow_empty and parsed.int == 0:
        raise ValueError(f"{field} must not be empty")
    return str(parsed)


def encode_message(message_type: str, request_id: str | UUID, payload: Payload) -> bytes:
    """Encode one MachiningMonitor-compatible length-prefixed JSON frame."""
    if not message_type or len(message_type) > 64:
        raise ValueError("message type must contain 1..64 characters")
    request = _uuid(request_id, "requestId")
    body = json.dumps(
        {
            "type": message_type,
            "requestId": request,
            "payload": _require_object(payload, "payload"),
            "version": PROTOCOL_VERSION,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(body) == 0 or len(body) > MAXIMUM_FRAME_BYTES:
        raise ValueError("TCP frame exceeds the 1 MiB maximum")
    return len(body).to_bytes(4, "big") + body


async def read_message(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read, size-check and validate one complete protocol frame."""
    header = await reader.readexactly(4)
    size = int.from_bytes(header, "big")
    if size == 0 or size > MAXIMUM_FRAME_BYTES:
        raise ValueError("invalid TCP frame length")
    raw = await reader.readexactly(size)
    try:
        message = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid UTF-8 JSON envelope") from exc
    message = _require_object(message, "message")
    if message.get("version") != PROTOCOL_VERSION:
        raise ValueError("unsupported TCP protocol version")
    message_type = message.get("type")
    if not isinstance(message_type, str) or not 0 < len(message_type) <= 64:
        raise ValueError("invalid message type")
    request_id = _uuid(message.get("requestId"), "requestId")
    payload = _require_object(message.get("payload"), "payload")
    return {"type": message_type, "requestId": request_id, "payload": payload, "version": PROTOCOL_VERSION}


def _validate_start(payload: Payload) -> Payload:
    result = dict(payload)
    _reject_unknown_fields(
        result,
        {"sessionId", "url", "username", "password", "machineId", "origin", "executionId", "leaseSeconds", "waitForAudio"},
        "analysis.start",
    )
    result["sessionId"] = _uuid(result.get("sessionId"), "sessionId")
    url = result.get("url")
    if not isinstance(url, str) or len(url) > 4096:
        raise ValueError("url must be a valid RTSP URL")
    parsed = urlsplit(url)
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("url contains an invalid port") from exc
    if parsed.scheme.lower() not in {"rtsp", "rtsps"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("url must be RTSP/RTSPS without embedded credentials")
    if not isinstance(result.get("username"), str) or not isinstance(result.get("password"), str):
        raise ValueError("username and password are required strings")
    if len(result["username"]) > 1024 or len(result["password"]) > 4096:
        raise ValueError("RTSP credentials exceed the allowed size")
    origin = result.get("origin", "Driver")
    if origin not in {"Driver", "RecordingManager"}:
        raise ValueError("origin is invalid")
    lease = result.get("leaseSeconds", 20)
    if isinstance(lease, bool) or not isinstance(lease, int) or not 5 <= lease <= 60:
        raise ValueError("leaseSeconds must be between 5 and 60")
    result["leaseSeconds"] = lease
    wait_for_audio = result.get("waitForAudio", False)
    if not isinstance(wait_for_audio, bool):
        raise ValueError("waitForAudio must be a boolean")
    result["waitForAudio"] = wait_for_audio
    if result.get("machineId") is not None:
        result["machineId"] = _uuid(result["machineId"], "machineId", allow_empty=True)
    if result.get("executionId") is not None and not isinstance(result["executionId"], str):
        raise ValueError("executionId must be a string or null")
    return result


def _validate_session_payload(
    payload: Payload,
    field: str = "sessionId",
    allowed_fields: set[str] | None = None,
) -> Payload:
    result = dict(payload)
    _reject_unknown_fields(result, allowed_fields or {field}, "analysis session command")
    result[field] = _uuid(result.get(field), field)
    return result


def _validate_data(payload: Payload) -> Payload:
    result = _validate_session_payload(
        payload,
        allowed_fields={"sessionId", "sequence", "kind", "data", "positionSeconds"},
    )
    sequence = result.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise ValueError("sequence must be a positive integer")
    if result.get("kind") not in {"context", "event", "playback"}:
        raise ValueError("kind is invalid")
    position = result.get("positionSeconds", 0)
    if isinstance(position, bool) or not isinstance(position, (int, float)) or not math.isfinite(position) or position < 0:
        raise ValueError("positionSeconds must be non-negative")
    if "data" not in result:
        raise ValueError("data is required")
    return result


def _reject_unknown_fields(payload: Payload, allowed: set[str], name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unsupported fields: {', '.join(unknown)}")


class MachiningMonitorSocketServer:
    """TLS Analyzer endpoint consumed by MachiningMonitor.

    The callbacks are intentionally application-level.  They must not block
    the event loop; use an executor or an async queue for model inference.
    The default handlers acknowledge the protocol without claiming that audio
    or model inference succeeded.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 7444,
        certificate_file: str | Path,
        private_key_file: str | Path | None = None,
        on_start: MessageHandler | None = None,
        on_data: MessageHandler | None = None,
        on_stop: MessageHandler | None = None,
        on_heartbeat: MessageHandler | None = None,
        on_audio: AudioChunkHandler | None = None,
        ffmpeg: str = DEFAULT_FFMPEG,
        audio_transport: str = "tcp",
        audio_chunk_ms: int = DEFAULT_AUDIO_CHUNK_MS,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        self.host = host
        self.port = port
        self.certificate_file = Path(certificate_file).expanduser().resolve()
        self.private_key_file = Path(private_key_file).expanduser().resolve() if private_key_file else None
        self._handlers = {
            "analysis.start": on_start,
            "analysis.data": on_data,
            "analysis.stop": on_stop,
            "analysis.heartbeat": on_heartbeat,
        }
        self._on_audio = on_audio
        self._ffmpeg = ffmpeg
        self._audio_transport = audio_transport
        self._audio_chunk_ms = audio_chunk_ms
        self._logger = logger or (lambda message: None)
        self._server: asyncio.AbstractServer | None = None
        self._connections: set[asyncio.Task[None]] = set()
        self._sessions: dict[str, _AnalysisSession] = {}
        self._ended_sessions: set[str] = set()

    def _ssl_context(self) -> ssl.SSLContext:
        if not self.certificate_file.is_file():
            raise FileNotFoundError(f"TLS certificate not found: {self.certificate_file}")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(str(self.certificate_file), str(self.private_key_file) if self.private_key_file else None)
        return context

    async def start(self) -> None:
        """Bind the TLS listener.  Raises if the port cannot be bound."""
        if self._server is not None:
            raise RuntimeError("server is already running")
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.host,
            port=self.port,
            ssl=self._ssl_context(),
            limit=MAXIMUM_FRAME_BYTES + 4,
        )
        self._logger(f"MachiningMonitor Analyzer socket listening on {self.host}:{self.port}")

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            await server.wait_closed()
        tasks = tuple(self._connections)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for session in tuple(self._sessions.values()):
            if session.audio is not None:
                await session.audio.stop()
        self._sessions.clear()
        self._ended_sessions.clear()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._connections.add(task)
        peer = writer.get_extra_info("peername")
        try:
            while True:
                message = await read_message(reader)
                response = await self._dispatch(message["type"], message["payload"])
                writer.write(encode_message(ACCEPTED_MESSAGE_TYPE, message["requestId"], response))
                await writer.drain()
        except asyncio.IncompleteReadError as exc:
            # A TLS client may probe the port and close before sending the
            # first protocol frame.  This is a normal disconnect, not an
            # unhandled server exception.  A partial frame is still reported
            # because it can indicate a protocol or transport failure.
            if exc.partial:
                self._logger(f"MachiningMonitor Analyzer socket ended with an incomplete frame for {peer}")
        except asyncio.CancelledError:
            raise
        except (ConnectionError, OSError, ValueError, json.JSONDecodeError) as exc:
            self._logger(f"MachiningMonitor Analyzer socket closed for {peer}: {exc}")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            if task is not None:
                self._connections.discard(task)

    async def _dispatch(self, message_type: str, payload: Payload) -> Payload:
        await self._expire_sessions()
        if message_type == "analysis.ping":
            return {"sessionId": str(UUID(int=0)), "state": "ready"}
        if message_type == "analysis.start":
            payload = _validate_start(payload)
            return await self._start_session(payload)
        elif message_type == "analysis.data":
            payload = _validate_data(payload)
            return await self._data_session(payload)
        elif message_type in {"analysis.stop", "analysis.heartbeat"}:
            payload = _validate_session_payload(payload)
            if message_type == "analysis.stop":
                return await self._stop_session(payload)
            return await self._heartbeat_session(payload)
        else:
            raise ValueError(f"unknown analyzer command: {message_type}")

    async def _call_handler(self, message_type: str, payload: Payload, default_state: str) -> Payload:
        handler = self._handlers[message_type]
        if handler is None:
            return {"sessionId": payload["sessionId"], "state": default_state}
        result = handler(payload)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, str):
            return {"sessionId": payload["sessionId"], "state": result}
        result = _require_object(result, "handler result")
        response = dict(result)
        response.setdefault("sessionId", payload["sessionId"])
        if not isinstance(response.get("state"), str) or not response["state"]:
            raise ValueError("handler result must contain a non-empty state")
        return response

    async def _start_session(self, payload: Payload) -> Payload:
        session_id = payload["sessionId"]
        existing = self._sessions.get(session_id)
        if existing is not None:
            if existing.start != payload:
                raise ValueError("session identity conflict")
            existing.lease_deadline = time.monotonic() + payload["leaseSeconds"]
            return {"sessionId": session_id, "state": existing.state}
        if session_id in self._ended_sessions:
            raise ValueError("session already ended; create a new session")
        if len(self._sessions) >= 128:
            raise ValueError("analysis session limit reached")
        response = await self._call_handler("analysis.start", payload, "accepted")
        state = response["state"]
        session = _AnalysisSession(
            start=dict(payload),
            state=state,
            lease_deadline=time.monotonic() + payload["leaseSeconds"],
        )
        session.audio = _AudioStreamWorker(
            session_id=session_id,
            url=payload["url"],
            username=payload["username"],
            password=payload["password"],
            ffmpeg=self._ffmpeg,
            transport=self._audio_transport,
            chunk_ms=self._audio_chunk_ms,
            context_provider=lambda: session.latest_context or {},
            on_audio=self._on_audio,
            logger=self._logger,
        )
        self._sessions[session_id] = session
        try:
            await session.audio.start()
            if payload["waitForAudio"]:
                await session.audio.wait_ready()
                response = {**response, "state": "audioReady"}
        except Exception:
            self._sessions.pop(session_id, None)
            await session.audio.stop()
            raise
        return response

    async def _data_session(self, payload: Payload) -> Payload:
        session_id = payload["sessionId"]
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError("unknown session")
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        sequence = payload["sequence"]
        if sequence < session.sequence or (sequence == session.sequence and serialized != session.last_data):
            raise ValueError("out of order or conflicting data")
        if sequence > session.sequence:
            session.sequence = sequence
            session.last_data = serialized
        data = payload["data"]
        if isinstance(data, dict):
            context = data.get("Context", data.get("context"))
            message = data.get("Message", data.get("message"))
            if isinstance(context, dict):
                session.latest_context = context
            if isinstance(message, dict):
                session.latest_message = message
        session.lease_deadline = time.monotonic() + session.start["leaseSeconds"]
        default_state = "dataAccepted" if session.audio and session.audio.is_receiving else "audioReconnecting"
        return await self._call_handler("analysis.data", payload, default_state)

    async def _heartbeat_session(self, payload: Payload) -> Payload:
        session = self._sessions.get(payload["sessionId"])
        if session is None:
            raise ValueError("unknown session")
        session.lease_deadline = time.monotonic() + session.start["leaseSeconds"]
        return await self._call_handler("analysis.heartbeat", payload, "alive")

    async def _stop_session(self, payload: Payload) -> Payload:
        session_id = payload["sessionId"]
        session = self._sessions.pop(session_id, None)
        try:
            return await self._call_handler("analysis.stop", payload, "stopped")
        finally:
            if session is not None and session.audio is not None:
                await session.audio.stop()
            self._ended_sessions.add(session_id)
            if len(self._ended_sessions) > 10000:
                self._ended_sessions.clear()

    async def _expire_sessions(self) -> None:
        now = time.monotonic()
        expired = [session_id for session_id, session in self._sessions.items() if session.lease_deadline <= now]
        for session_id in expired:
            try:
                await self._stop_session({"sessionId": session_id})
            except Exception as exc:
                self._logger(f"Analyzer session {session_id} expired: {exc}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MachiningMonitor Analyzer TLS socket server")
    parser.add_argument("--host", default="127.0.0.1", help="Local address to bind")
    parser.add_argument("--port", type=int, default=7444, help="TLS socket port")
    parser.add_argument("--ffmpeg", default=DEFAULT_FFMPEG, help="FFmpeg executable used for each RTSP session")
    parser.add_argument("--audio-transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--audio-chunk-ms", type=int, default=DEFAULT_AUDIO_CHUNK_MS)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "rtsp" / "web_sessions",
        help="Directory where socket audio/context sessions are persisted",
    )
    parser.add_argument(
        "--certificate",
        type=Path,
        default=DEFAULT_CERTIFICATE,
        help=f"PEM certificate file (default: {DEFAULT_CERTIFICATE})",
    )
    parser.add_argument(
        "--private-key",
        type=Path,
        default=DEFAULT_PRIVATE_KEY,
        help=f"PEM private key file (default: {DEFAULT_PRIVATE_KEY})",
    )
    return parser.parse_args(argv)


async def _run_server(args: argparse.Namespace) -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Integration.socket_training_bridge import SocketTrainingBridge

    bridge = SocketTrainingBridge(args.training_root, print, transport=args.audio_transport)
    server = MachiningMonitorSocketServer(
        host=args.host,
        port=args.port,
        certificate_file=args.certificate,
        private_key_file=args.private_key,
        ffmpeg=args.ffmpeg,
        audio_transport=args.audio_transport,
        audio_chunk_ms=args.audio_chunk_ms,
        on_start=bridge.on_start,
        on_data=bridge.on_data,
        on_stop=bridge.on_stop,
        on_audio=bridge.on_audio,
        logger=print,
    )
    try:
        await server.serve_forever()
    finally:
        await server.stop()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        asyncio.run(_run_server(args))
    except KeyboardInterrupt:
        print("MachiningMonitor Analyzer socket stopped.")
    except (FileNotFoundError, OSError, ssl.SSLError) as exc:
        print(
            "No se pudo iniciar el servidor TLS: " + str(exc) +
            "\nGenere el certificado con: "
            ".\\Integration\\New-AnalyzerDevCertificate.ps1",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

