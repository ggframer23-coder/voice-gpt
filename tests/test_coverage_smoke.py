import stt.cli as cli


def test_coverage_smoke() -> None:
    assert hasattr(cli, "_resolve_engine_for_audio")
