from modes.talkback.gbnf import SPEECH_GRAMMAR


def test_grammar_excludes_markdown_markers():
    for ch in "*#`_~":
        assert ch in SPEECH_GRAMMAR  # present in the negated class

def test_grammar_is_a_single_root_rule():
    assert SPEECH_GRAMMAR.startswith("root ::=")

def test_grammar_excludes_fullwidth_homoglyphs():
    # U+FF0A FULLWIDTH ASTERISK, U+FF03 FULLWIDTH NUMBER SIGN,
    # U+FF40 FULLWIDTH GRAVE ACCENT, U+FF3F FULLWIDTH LOW LINE,
    # U+FF5E FULLWIDTH TILDE — homoglyphs the decoder used to route
    # markdown through once the ASCII markers were banned.
    for ch in "＊＃｀＿～":
        assert ch in SPEECH_GRAMMAR
