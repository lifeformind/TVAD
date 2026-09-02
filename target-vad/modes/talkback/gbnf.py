"""GBNF grammar for llama-server: make markdown structurally impossible.

The decoder simply cannot emit *, #, backtick, _ or ~ — the characters
strip_markdown_for_speech (speech_text.py) spends most of its regexes on.
The stripper stays as belt-and-suspenders for what a char class can't
express (bullet dashes at line starts, [link](url) syntax).
"""

# Any character except markdown marker characters; explicitly allows
# newlines, punctuation, and unicode (GBNF negated classes are by
# codepoint, so non-ASCII passes).
SPEECH_GRAMMAR = 'root ::= [^*#`_~]*'
