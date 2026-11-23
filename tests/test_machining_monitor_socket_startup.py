from pathlib import Path

from Integration.machining_monitor_socket import _parse_args


def test_socket_server_cli_requires_tls_files_and_defaults_to_analyzer_port():
    args = _parse_args(
        [
            "--certificate",
            "certs/analyzer.crt",
            "--private-key",
            "certs/analyzer.key",
        ]
    )

    assert args.host == "127.0.0.1"
    assert args.port == 7444
    assert args.certificate == Path("certs/analyzer.crt")
    assert args.private_key == Path("certs/analyzer.key")


def test_socket_server_cli_has_development_tls_defaults():
    args = _parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 7444
    assert args.certificate.name == "machining-assistant-analyzer.crt"
    assert args.private_key.name == "machining-assistant-analyzer.key"
