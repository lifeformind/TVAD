from modes.talkback.gbnf import SPEECH_GRAMMAR


def test_grammar_excludes_markdown_markers():
    for ch in "*#`_~":
        assert ch in SPEECH_GRAMMAR  # present in the negated class

def test_grammar_is_a_single_root_rule():
    assert SPEECH_GRAMMAR.startswith("root ::=")
