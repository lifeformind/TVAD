"""GBNF grammar for llama-server: make markdown structurally impossible.

The decoder simply cannot emit *, #, backtick, _ or ~ — the characters
strip_markdown_for_speech (speech_text.py) spends most of its regexes on.
The stripper stays as belt-and-suspenders for what a char class can't
express (bullet dashes at line starts, [link](url) syntax).

Also bans the fullwidth homoglyphs ＊＃｀＿～ (U+FF0A, U+FF03, U+FF40,
U+FF3F, U+FF5E): once the ASCII markers were banned, the decoder was
observed routing markdown through these visually-similar codepoints
instead — GBNF negated classes are by codepoint, so each lookalike must
be listed explicitly rather than relying on the ASCII ban alone.
"""

# Any character except markdown marker characters (ASCII and their
# fullwidth homoglyphs); explicitly allows newlines, punctuation, and
# other unicode (GBNF negated classes are by codepoint, so anything not
# listed passes).
SPEECH_GRAMMAR = 'root ::= [^*#`_~＊＃｀＿～]*'
