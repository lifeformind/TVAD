"""Pure backchannel-vs-question classifier for barge-in interjections.

Maps a transcript to BACKCHANNEL (the user is just acknowledging — keep
talking) or INTERRUPT (a genuine question/command — cut and respond). No I/O,
no state; the SPEAKING-branch barge-in path calls classify_interjection() on
the (already speaker-verified) interjection transcript BEFORE deciding to cut.

Design (review rec R2):
- Stage 0 EMPTY/GARBAGE GUARD (mandatory): empty / punctuation-only / known
  whisper-tiny hallucinations -> BACKCHANNEL. Never cut on garbage STT.
- Stage 1 LEXICAL: any FORCE_INTERRUPT token wins -> INTERRUPT; else if every
  token is a known backchannel -> BACKCHANNEL; else (real content) -> INTERRUPT
  (default-to-cut; a future auto-resume net self-heals wrong cuts).
"""

import enum
import re


class Interjection(enum.Enum):
    BACKCHANNEL = "BACKCHANNEL"
    INTERRUPT = "INTERRUPT"


# A force token anywhere forces a cut (questions / commands / repair).
FORCE_INTERRUPT = {
    "why", "what", "how", "when", "where", "who", "which",
    "wait", "stop", "no", "hold", "pause", "sorry", "repeat", "again",
    "but", "actually", "pardon", "explain", "mean",
}

# If EVERY token is here (and none are force tokens), it's an acknowledgment.
# Includes common whisper-tiny near-silence hallucinations (you/thanks/i).
BACKCHANNEL = {
    "okay", "ok", "yeah", "yep", "yup", "yes", "uhhuh", "mhm", "mm", "hmm",
    "right", "sure", "got", "it", "gotcha", "cool", "nice", "wow", "oh", "ah",
    "totally", "exactly", "makes", "sense", "fair", "true", "go", "on",
    "continue", "i", "see", "you", "thanks", "thank", "alright", "great",
}


def _tokenize(text: str) -> list[str]:
    t = text.lower()
    # normalize multi-word backchannels to single known tokens
    t = (t.replace("uh-huh", "uhhuh").replace("uh huh", "uhhuh")
          .replace("mm-hmm", "mhm").replace("mm hmm", "mhm")
          .replace("mhmm", "mhm"))
    return re.findall(r"[a-z]+", t)


def classify_interjection(text: str) -> Interjection:
    tokens = _tokenize(text or "")
    if not tokens:                                    # Stage 0: garbage guard
        return Interjection.BACKCHANNEL
    if any(tok in FORCE_INTERRUPT for tok in tokens):  # Stage 1: force wins
        return Interjection.INTERRUPT
    if all(tok in BACKCHANNEL for tok in tokens):      # all-acknowledgment
        return Interjection.BACKCHANNEL
    return Interjection.INTERRUPT                       # default-to-cut
