from lyricpredict.confidence import ConfidenceGate, ConfidenceSettings


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
