import sys

from wostrategy.script import tyre_prediction_pipeline as cli


def test_processed_mode_never_uses_ambient_vision_setting(monkeypatch, capsys):
    monkeypatch.setenv("WODATA_PIRELLI_VISION_MODEL", "ambient-vision-model")
    monkeypatch.setattr(sys, "argv", [
        "tyre_prediction_pipeline", "--year", "2026", "--processed-previews-only",
        "--no-refresh-fp-calibration",
    ])
    calls = []

    def unexpected_extractor(**kwargs):
        raise AssertionError("processed input must never construct a vision extractor")

    def pipeline(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(cli, "OpenAIPreviewVisionExtractor", unexpected_extractor)
    monkeypatch.setattr(cli, "run_tyre_prediction_pipeline", pipeline)
    monkeypatch.setattr(cli, "_report_dict", lambda report: {"completed": True})
    cli.main()
    assert calls[0]["vision_extractor"] is None
    assert calls[0]["processed_previews_only"] is True
    assert calls[0]["refresh_fp_calibration"] is False
    assert "completed" in capsys.readouterr().out
