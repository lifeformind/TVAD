"""strip_markdown_for_speech — remove markdown markup before TTS synthesis.

gemma (and LLMs generally) emit markdown emphasis — *italic*, **bold**, `code`,
[links](url), # headers, - bullets — even when the system prompt asks them not
to. A voice assistant must never vocalize those markers (Kokoro will literally
speak the asterisks). This strips them deterministically; a prompt instruction
alone is unreliable.

Unwraps the common inline patterns (keeping the inner words), drops leading
header/bullet markers, then removes any leftover markup characters — including a
lone marker left when emphasis is split across streamed sentence chunks. Leaves
ordinary prose (apostrophes, hyphens, en/em dashes) untouched.
"""

import re

_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")     # ![alt](url) -> alt
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")       # [text](url) -> text
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")             # **bold** -> bold
_MD_BOLD_U = re.compile(r"__([^_]+)__")               # __bold__ -> bold
_MD_STRIKE = re.compile(r"~~([^~]+)~~")               # ~~x~~ -> x
_MD_CODE = re.compile(r"`([^`]+)`")                   # `code` -> code
_MD_ITALIC = re.compile(r"\*([^*]+)\*")               # *italic* -> italic
_MD_ITALIC_U = re.compile(r"_([^_]+)_")               # _italic_ -> italic
_MD_HEADER = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")     # leading ATX header marker
_MD_BULLET = re.compile(r"(?m)^\s{0,3}[-*+]\s+")      # leading list bullet
_MD_STRAY = re.compile(r"[*`_~#]")                    # leftover markup chars
_WS = re.compile(r"\s{2,}")


def strip_markdown_for_speech(text: str) -> str:
    text = _MD_IMAGE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_BULLET.sub("", text)
    text = _MD_BOLD.sub(r"\1", text)          # bold before italic (** before *)
    text = _MD_BOLD_U.sub(r"\1", text)
    text = _MD_STRIKE.sub(r"\1", text)
    text = _MD_CODE.sub(r"\1", text)
    text = _MD_ITALIC.sub(r"\1", text)
    text = _MD_ITALIC_U.sub(r"\1", text)
    text = _MD_STRAY.sub("", text)            # lone markers (split across chunks)
    text = _WS.sub(" ", text)
    return text.strip()
