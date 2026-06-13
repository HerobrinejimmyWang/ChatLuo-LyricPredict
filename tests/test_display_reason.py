from lyricpredict.desktop.workflow import display_reason


def test_display_reason_hides_internal_pipeline_names():
    assert display_reason("char_match_suffix") == "matched"
    assert display_reason("char_match_half_exact") == "partial matched"
    assert display_reason("char_match_ambiguous") == "ambiguous match"
    assert display_reason("char_match_threshold") == "low confidence"
    assert display_reason("retrieval") == "matched"
    assert display_reason("verified_transformer:ngram_exact") == "verified"
    assert display_reason("verified_transformer:ngram_fuzzy") == "verified with correction"
    assert display_reason("low_final_confidence") == "low confidence"
