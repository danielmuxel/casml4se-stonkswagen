from __future__ import annotations

from datetime import UTC, datetime

from gw2ml.modeling import load_model, persist_model


def test_persist_and_load_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GW2ML_MODELS_DIR", str(tmp_path))
    model_object = {"coef": [1, 2, 3]}
    artifact_path = persist_model(model_object, model_key="demo", run_date=datetime(2025, 1, 1, tzinfo=UTC))
    assert artifact_path.exists()

    loaded = load_model("demo")
    assert loaded == model_object




