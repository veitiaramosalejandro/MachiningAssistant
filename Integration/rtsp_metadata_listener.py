"""Read the x-machining-metadata RTP track from the local RTSP publisher.

The publisher sends MAM1 frames over RTSP interleaved TCP channel 2:
ASCII ``MAM1``, a big-endian UInt32 JSON length, and UTF-8 JSON bytes.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import socket
import time
from urllib.parse import unquote, urlsplit


MAGIC = b"MAM1"
MAX_FRAME_BYTES = 8 * 1024 * 1024


def _safe_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme.lower() != "rtsp" or not parts.hostname:
        return "rtsp://<redacted>"
    host = parts.hostname
    if parts.port:
        host += f":{parts.port}"
    return f"rtsp://{host}{parts.path or '/'}"


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class RtspMetadataClient:
    def __init__(self, url: str, output: Path, status_output: Path) -> None:
        self.url = url
        self.parts = urlsplit(url)
        if self.parts.scheme.lower() != "rtsp" or not self.parts.hostname:
            raise ValueError("MACHINING_METADATA_RTSP_URL debe ser una URL RTSP válida")
        self.output = output
        self.status_output = status_output
        self.socket: socket.socket | None = None
        self.cseq = 1
        self.session: str | None = None
        self.buffer = bytearray()
        self.status = {
            "status": "starting", "rtsp_url": _safe_url(url), "track": "x-machining-metadata",
            "framing": "MAM1 + big-endian UInt32 JSON length + UTF-8 JSON",
            "rtp_channel": 2, "packets_received": 0, "bytes_received": 0,
            "messages_received": 0, "message_types": {}, "events_received": 0,
            "snapshots_received": 0, "last_message_type": None, "last_timestamp_seconds": None,
            "last_sequence": None, "error": None,
        }

    def _authorization(self) -> str | None:
        if self.parts.username is None or self.parts.password is None:
            return None
        # URL userinfo may contain percent-encoded characters (for example
        # an asterisk in the password); Basic auth needs the original text.
        username = unquote(self.parts.username)
        password = unquote(self.parts.password)
        raw = f"{username}:{password}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    def _request(self, method: str, uri: str, extra: dict[str, str] | None = None) -> tuple[str, dict[str, str]]:
        headers = {"CSeq": str(self.cseq), "User-Agent": "MachiningAssistantMetadata/1.0"}
        self.cseq += 1
        if self.session:
            headers["Session"] = self.session
        if extra:
            headers.update(extra)
        authorization = self._authorization()
        if authorization:
            headers["Authorization"] = authorization
        request = method + " " + uri + " RTSP/1.0\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n"
        assert self.socket is not None
        self.socket.sendall(request.encode("ascii"))
        response = self._read_rtsp_response()
        if response[0].startswith("RTSP/1.0 401") and authorization is None:
            raise RuntimeError("El stream RTSP requiere usuario y contraseña")
        if not response[0].startswith("RTSP/1.0 2"):
            raise RuntimeError(f"RTSP {method} rechazado: {response[0]}")
        return response

    def _read_rtsp_response(self) -> tuple[str, dict[str, str]]:
        assert self.socket is not None
        data = bytearray()
        while b"\r\n\r\n" not in data:
            chunk = self.socket.recv(4096)
            if not chunk:
                raise ConnectionError("El servidor RTSP cerró la conexión durante la respuesta")
            data.extend(chunk)
            if len(data) > 128 * 1024:
                raise RuntimeError("Respuesta RTSP demasiado grande")
        head, _, body = bytes(data).partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        content_length = int(headers.get("content-length", "0") or 0)
        while len(body) < content_length:
            body += self.socket.recv(content_length - len(body))
        return lines[0], headers

    def _connect(self) -> None:
        port = self.parts.port or 554
        self.socket = socket.create_connection((self.parts.hostname, port), timeout=15)
        self.socket.settimeout(2.0)
        base = f"rtsp://{self.parts.hostname}:{port}{self.parts.path or '/'}"
        self._request("OPTIONS", base)
        status, headers = self._request("DESCRIBE", base, {"Accept": "application/sdp"})
        _ = status
        # The custom server accepts trackID=1 for the metadata application track.
        track = base.rstrip("/") + "/trackID=1"
        _, setup_headers = self._request("SETUP", track, {"Transport": "RTP/AVP/TCP;unicast;interleaved=2-3"})
        self.session = setup_headers.get("session", "").split(";", 1)[0] or None
        self._request("PLAY", base)

    def _consume_rtp(self, packet: bytes) -> None:
        if len(packet) < 12 or (packet[0] >> 6) != 2:
            return
        header_length = 12 + 4 * (packet[0] & 0x0F)
        if packet[0] & 0x10:
            if len(packet) < header_length + 4:
                return
            extension_words = int.from_bytes(packet[header_length + 2:header_length + 4], "big")
            header_length += 4 + extension_words * 4
        if header_length > len(packet):
            return
        payload = packet[header_length:]
        if packet[0] & 0x20 and payload:
            payload = payload[:-payload[-1]]
        self.buffer.extend(payload)
        self._extract_messages()

    def _extract_messages(self) -> None:
        while True:
            if len(self.buffer) < 8:
                return
            if self.buffer[:4] != MAGIC:
                index = self.buffer.find(MAGIC, 1)
                if index < 0:
                    del self.buffer[:-3]
                    return
                del self.buffer[:index]
            length = int.from_bytes(self.buffer[4:8], "big")
            if length > MAX_FRAME_BYTES:
                del self.buffer[:4]
                continue
            if len(self.buffer) < 8 + length:
                return
            raw = bytes(self.buffer[8:8 + length])
            del self.buffer[:8 + length]
            try:
                message = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(message, dict):
                continue
            with self.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(message, ensure_ascii=False) + "\n")
            message_type = str(message.get("type", "unknown"))
            self.status["messages_received"] += 1
            self.status["message_types"][message_type] = self.status["message_types"].get(message_type, 0) + 1
            self.status["last_message_type"] = message_type
            self.status["last_timestamp_seconds"] = message.get("timestampSeconds")
            self.status["last_sequence"] = message.get("sequence")
            if message_type == "event":
                self.status["events_received"] += 1
            if message_type == "snapshot":
                self.status["snapshots_received"] += 1
            _write_json(self.status_output, self.status)

    def run(self) -> int:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.status_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connect()
            self.status["status"] = "running"
            _write_json(self.status_output, self.status)
            assert self.socket is not None
            while True:
                try:
                    first = self.socket.recv(1)
                except socket.timeout:
                    continue
                if not first:
                    raise ConnectionError("El servidor RTSP cerró la conexión")
                if first != b"$":
                    # RTSP control responses are not expected after PLAY; discard the line safely.
                    continue
                header = self.socket.recv(3)
                if len(header) != 3:
                    raise ConnectionError("Trama RTP intercalada incompleta")
                channel = header[0]
                length = int.from_bytes(header[1:3], "big")
                packet = self.socket.recv(length)
                if len(packet) != length:
                    raise ConnectionError("Paquete RTP intercalado incompleto")
                self.status["packets_received"] += 1
                self.status["bytes_received"] += length
                if channel == 2:
                    self._consume_rtp(packet)
                if self.status["packets_received"] % 50 == 0:
                    _write_json(self.status_output, self.status)
        except Exception as exc:
            self.status.update({"status": "failed", "error": str(exc)})
            _write_json(self.status_output, self.status)
            return 1
        finally:
            if self.socket is not None:
                self.socket.close()


def main() -> int:
    url = os.environ.get("MACHINING_METADATA_RTSP_URL", "")
    output = Path(os.environ.get("MACHINING_METADATA_OUTPUT", "runs/rtsp/metadata.jsonl"))
    status_output = Path(os.environ.get("MACHINING_METADATA_STATUS_OUTPUT", "runs/rtsp/metadata.status.json"))
    if not url:
        raise SystemExit("MACHINING_METADATA_RTSP_URL is required")
    return RtspMetadataClient(url, output, status_output).run()


if __name__ == "__main__":
    raise SystemExit(main())
