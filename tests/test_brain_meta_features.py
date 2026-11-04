import pytest
from brain.meta_features import ContentSpan, ContentFact, SemanticResult, validate_semantics, restore_source_whitespace


def test_semantics_requires_verbatim_source_evidence_and_known_span():
    spans = [ContentSpan(id="s1", start=0, stop=10, text="Velocity is displacement per unit time.")]
    valid = SemanticResult(facts=[ContentFact(span_id="s1", kind="definition", quote="Velocity is displacement")], sam_nouns=[])
    validate_semantics(valid, spans)
    with pytest.raises(ValueError, match="exact quote"):
        validate_semantics(SemanticResult(facts=[ContentFact(span_id="s1", kind="definition", quote="The student is attentive")], sam_nouns=[]), spans)
    with pytest.raises(ValueError, match="exact quote"):
        validate_semantics(SemanticResult(facts=[ContentFact(span_id="missing", kind="definition", quote="Velocity")], sam_nouns=[]), spans)


def test_content_spans_reject_nonfinite_or_reversed_timing():
    for start, stop in [(10, 1), (0, float("inf")), (float("nan"), 10)]:
        with pytest.raises(ValueError):
            ContentSpan(id="s", start=start, stop=stop, text="test")


def test_pdf_line_wrapping_restores_exact_source_without_changing_raw_result():
    spans = [ContentSpan(id="p", start=0, stop=32, text="The cart travels\nten meters east.")]
    raw = SemanticResult(facts=[ContentFact(span_id="p", kind="worked_example",
        quote="The cart travels ten meters east.")], sam_nouns=[])
    grounded, repairs = restore_source_whitespace(raw, spans)
    assert grounded.facts[0].quote == spans[0].text
    assert raw.facts[0].quote == "The cart travels ten meters east."
    assert repairs[0]["source_quote"] == spans[0].text
    validate_semantics(grounded, spans)
    assert restore_source_whitespace(grounded, spans)[1] == []


@pytest.mark.parametrize("text,quote", [
    ("The cart travels\nten meters east.", "The cart travels twenty meters east."),
    ("The cart travels\nten meters east.", "The cart travels ten meters east!"),
    ("ten\nmeters and ten\tmeters", "ten meters"),
])
def test_whitespace_repair_rejects_changed_facts_and_ambiguous_matches(text, quote):
    spans = [ContentSpan(id="p", start=0, stop=32, text=text)]
    result = SemanticResult(facts=[ContentFact(span_id="p", kind="worked_example", quote=quote)], sam_nouns=[])
    with pytest.raises(ValueError, match="exact quote"):
        restore_source_whitespace(result, spans)
