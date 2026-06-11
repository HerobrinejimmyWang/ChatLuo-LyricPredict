import json

from lyricpredict.confidence import ConfidenceGate, ConfidenceSettings, load_confidence_profiles, save_confidence_profiles


def test_low_confidence_rejects_with_empty_output_contract():
    gate = ConfidenceGate(ConfidenceSettings(threshold=0.5, min_token_probability=0.01, max_repeat_ratio=0.9))
    result = gate.evaluate("一句。", [0.1, 0.1, 0.1], ended=True)
    assert not result.accepted
    assert result.reason == "threshold"


def test_no_terminator_rejects():
    gate = ConfidenceGate(ConfidenceSettings())
    result = gate.evaluate("一句", [0.9, 0.9], ended=False)
    assert not result.accepted
    assert result.reason == "no_terminator"


def test_good_candidate_accepts():
    gate = ConfidenceGate(ConfidenceSettings(threshold=0.2, min_token_probability=0.01, max_repeat_ratio=0.9))
    result = gate.evaluate("一句歌词。", [0.7, 0.7, 0.7, 0.7], ended=True)
    assert result.accepted


def test_hyphen_artifact_rejects():
    gate = ConfidenceGate(ConfidenceSettings(threshold=0.01, min_token_probability=0.0, max_repeat_ratio=0.9))
    result = gate.evaluate("- - - - 那麼風塵，", [0.8] * 8, ended=True)

    assert not result.accepted
    assert result.reason == "hyphen_artifact"


def test_legacy_confidence_file_keeps_ngram_default(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "confidence.json").write_text(
        json.dumps({"threshold": 0.92, "min_token_probability": 0.0, "max_repeat_ratio": 0.35}),
        encoding="utf-8",
    )

    profiles = load_confidence_profiles(model_dir, ConfidenceSettings(threshold=0.18))

    assert profiles.legacy is not None
    assert profiles.profile("retrieval").threshold == 0.92
    assert profiles.profile("ngram").threshold == 0.05
    assert profiles.profile("transformer").threshold == 0.18


def test_v2_confidence_file_loads_profiles_independently(tmp_path):
    model_dir = tmp_path / "model"
    save_confidence_profiles(
        model_dir,
        {
            "retrieval": ConfidenceSettings(threshold=0.91, min_token_probability=0.0, max_repeat_ratio=0.3),
            "ngram": ConfidenceSettings(threshold=0.04, min_token_probability=0.0, max_repeat_ratio=0.5),
            "transformer": ConfidenceSettings(threshold=0.2, min_token_probability=0.01, max_repeat_ratio=0.4),
        },
        samples={"ngram": [0.1]},
    )

    profiles = load_confidence_profiles(model_dir, ConfidenceSettings())

    assert profiles.profile("retrieval").threshold == 0.91
    assert profiles.profile("ngram").threshold == 0.04
    assert profiles.profile("transformer").min_token_probability == 0.01
    assert profiles.samples == {"retrieval": [], "ngram": [0.1], "transformer": []}
