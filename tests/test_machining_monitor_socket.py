import asyncio
import json
from uuid import uuid4

import pytest

from Integration.machining_monitor_socket import (
    MachiningMonitorSocketServer,
    encode_message,
    read_message,
)


def test_encode_message_uses_big_endian_length_and_protocol_envelope():
    request_id = uuid4()
    frame = encode_message("analysis.ping", request_id, {})
    size = int.from_bytes(frame[:4], "big")
    body = json.loads(frame[4:].decode("utf-8"))

    assert size == len(frame) - 4
    assert body == {
        "type": "analysis.ping",
        "requestId": str(request_id),
        "payload": {},
        "version": 1,
    }


def test_read_message_rejects_oversized_frame():
    async def scenario():
        reader = asyncio.StreamReader()
        reader.feed_data((1024 * 1024 + 1).to_bytes(4, "big"))
        reader.feed_eof()
        with pytest.raises(ValueError, match="invalid TCP frame length"):
            await read_message(reader)

    asyncio.run(scenario())


def test_dispatch_validates_start_and_returns_accepted_state():
    async def scenario():
        session_id = str(uuid4())
        server = MachiningMonitorSocketServer(
            certificate_file="unused-for-dispatch-test.pem",
            on_start=lambda payload: {"state": "audioReady", "machineId": payload.get("machineId")},
        )
        result = await server._dispatch(
            "analysis.start",
            {
                "sessionId": session_id,
                "url": "rtsp://127.0.0.1:8555/audio",
                "username": "audio",
                "password": "secret",
                "origin": "Driver",
                "leaseSeconds": 20,
            },
        )
        assert result == {"state": "audioReady", "machineId": None, "sessionId": session_id}

    asyncio.run(scenario())


def test_dispatch_rejects_embedded_rtsp_credentials():
    async def scenario():
        server = MachiningMonitorSocketServer(certificate_file="unused-for-dispatch-test.pem")
        with pytest.raises(ValueError, match="without embedded credentials"):
            await server._dispatch(
                "analysis.start",
                {
                    "sessionId": str(uuid4()),
                    "url": "rtsp://user:password@127.0.0.1:8555/audio",
                    "username": "audio",
                    "password": "secret",
                },
            )

    asyncio.run(scenario())

