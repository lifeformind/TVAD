#!/usr/bin/env python3
"""Speaker-verification score report from kiosk JSONL logs (Batch 2c harness).

Extracts every speaker-similarity cosine the kiosk records and summarizes the
distribution, so a "does the threshold reject a non-self voice?" run can be
characterized after the fact. The relevant events:

  turn_gate          {"score", "threshold", "decision"}  NEW turn vs primary (clean, no TTS)
  barge_in_rejected  {"score", "threshold"}        voice over TTS, score < threshold -> rejected
  barge_in           {"primary_score", ...}         voice over TTS, score >= threshold -> accepted
  segment_scored     {"score", "decision", ...}     kiosk primary-lock match (console-only today)

The `turn_gate` source is the cleanest non-self signal: it scores every new turn
in LISTENING state (no TTS playing to corrupt the embedding), unlike barge_in
which fights echo during playback.

IMPORTANT: a score is self-vs-self unless a *different* person produced it.
The tool cannot infer that — group by session (--group-by session) and run a
clean "second person speaks" session so its scores are isolated. Label a run
with --label to tag the printed report (e.g. --label nonself).

Usage:
  python3 bench/speaker_scores.py                         # all logs/kiosk-*.jsonl
  python3 bench/speaker_scores.py logs/kiosk-2026-06-09-*.jsonl
  python3 bench/speaker_scores.py --group-by session      # per-session breakdown
  python3 bench/speaker_scores.py --session d678dac57bc1 --label nonself
  python3 bench/speaker_scores.py --source barge_in --threshold 0.75
  # labeled self-vs-non-self separation (see docs/speaker-gate-measurement.md):
  python3 bench/speaker_scores.py --source turn_gate --self AAA --nonself BBB
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

DEFAULT_GLOB = "logs/kiosk-*.jsonl"
# thresholds used when the event payload doesn't carry one
DEFAULT_THRESHOLDS = {"turn_gate": 0.30, "barge_in": 0.75, "primary_match": 0.50}


@dataclass
class ScoreRow:
    session_id: str
    ts: str
    source: str          # "barge_in" | "primary_match"
    event: str           # raw event name
    score: float
    threshold: Optional[float]
    decision: str        # "accept" | "reject"


def iter_records(paths: Iterable[str]) -> Iterator[dict]:
    """Yield parsed JSON records from the given log files (bad lines skipped)."""
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def extract_scores(records: Iterable[dict]) -> list[ScoreRow]:
    """Pull every speaker-similarity score out of a stream of log records."""
    rows: list[ScoreRow] = []
    for r in records:
        event = r.get("event", "")
        payload = r.get("payload", {}) or {}
        sid = r.get("session_id", "?")
        ts = r.get("ts", "")
        if event == "turn_gate":
            decision = "accept" if payload.get("decision") == "accept" else "reject"
            rows.append(ScoreRow(sid, ts, "turn_gate", event,
                                 float(payload["score"]),
                                 _maybe_float(payload.get("threshold")), decision))
        elif event == "barge_in_rejected":
            rows.append(ScoreRow(sid, ts, "barge_in", event,
                                 float(payload["score"]),
                                 _maybe_float(payload.get("threshold")), "reject"))
        elif event == "barge_in":
            rows.append(ScoreRow(sid, ts, "barge_in", event,
                                 float(payload["primary_score"]),
                                 _maybe_float(payload.get("threshold")), "accept"))
        elif event == "segment_scored":
            decision = "accept" if payload.get("decision") == "match" else "reject"
            rows.append(ScoreRow(sid, ts, "primary_match", event,
                                 float(payload["score"]),
                                 _maybe_float(payload.get("threshold")), decision))
    return rows


def _maybe_float(v) -> Optional[float]:
    return None if v is None else float(v)


def summarize(scores: list[float]) -> dict:
    """Distribution stats for a list of cosine scores."""
    if not scores:
        return {"n": 0}
    s = sorted(scores)
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "median": statistics.median(s),
        "min": s[0],
        "max": s[-1],
        "stdev": statistics.pstdev(s) if len(s) > 1 else 0.0,
        "p10": _pct(s, 0.10),
        "p90": _pct(s, 0.90),
    }


def _pct(sorted_scores: list[float], q: float) -> float:
    if len(sorted_scores) == 1:
        return sorted_scores[0]
    idx = q * (len(sorted_scores) - 1)
    lo = int(idx)
    frac = idx - lo
    hi = min(lo + 1, len(sorted_scores) - 1)
    return sorted_scores[lo] * (1 - frac) + sorted_scores[hi] * frac


def histogram(scores: list[float], bins: int = 20, width: int = 40) -> str:
    """ASCII histogram of scores over [0, 1]."""
    if not scores:
        return "(no scores)"
    counts = [0] * bins
    for v in scores:
        b = min(int(max(0.0, min(1.0, v)) * bins), bins - 1)
        counts[b] += 1
    peak = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        lo, hi = i / bins, (i + 1) / bins
        bar = "#" * round(c / peak * width)
        lines.append(f"  {lo:4.2f}-{hi:4.2f} | {bar} {c}" if c else f"  {lo:4.2f}-{hi:4.2f} |")
    return "\n".join(lines)


def _accept_reject(rows: list[ScoreRow], threshold: float) -> tuple[int, int]:
    acc = sum(1 for r in rows if r.score >= threshold)
    return acc, len(rows) - acc


def best_threshold(self_scores: list[float],
                   nonself_scores: list[float]) -> dict:
    """Find the accept/reject threshold that best separates two labeled groups.

    A turn is accepted when score >= threshold. We sweep candidate thresholds
    (midpoints between observed scores) and pick the one with the highest
    balanced accuracy = (true-accept rate + true-reject rate) / 2.

    Returns {} if either group is empty. `margin` is min(self) - max(nonself):
    positive means the two groups are perfectly separable.
    """
    if not self_scores or not nonself_scores:
        return {}
    vals = sorted(set(self_scores) | set(nonself_scores))
    cands = [vals[0] - 0.01]
    for a, b in zip(vals, vals[1:]):
        cands.append((a + b) / 2.0)
    cands.append(vals[-1] + 0.01)

    n_self, n_non = len(self_scores), len(nonself_scores)
    best = None
    for t in cands:
        tpr = sum(s >= t for s in self_scores) / n_self          # accept self
        tnr = sum(s < t for s in nonself_scores) / n_non         # reject non-self
        bal = (tpr + tnr) / 2.0
        if best is None or bal > best["balanced_acc"]:
            best = {"threshold": t, "tpr": tpr, "tnr": tnr, "balanced_acc": bal}
    best["margin"] = min(self_scores) - max(nonself_scores)
    best["separable"] = best["margin"] > 0
    return best


def report_comparison(self_rows: list[ScoreRow], nonself_rows: list[ScoreRow],
                      bins: int) -> str:
    """Side-by-side self vs non-self distribution + best-separating threshold."""
    out = ["=== self vs non-self separation ==="]
    self_scores = [r.score for r in self_rows]
    nonself_scores = [r.score for r in nonself_rows]
    for name, scores in (("self (enrolled)", self_scores),
                         ("non-self (other)", nonself_scores)):
        st = summarize(scores)
        if st["n"] == 0:
            out.append(f"\n--- {name} ---\n  (no scores)")
            continue
        out.append(f"\n--- {name} ---")
        out.append(f"  n={st['n']}  mean={st['mean']:.3f}  median={st['median']:.3f}  "
                   f"stdev={st['stdev']:.3f}")
        out.append(f"  range=[{st['min']:.3f}, {st['max']:.3f}]  "
                   f"p10={st['p10']:.3f}  p90={st['p90']:.3f}")
        out.append(histogram(scores, bins=bins))

    bt = best_threshold(self_scores, nonself_scores)
    out.append("\n--- separation ---")
    if not bt:
        out.append("  Need both a self and a non-self group to compare.")
        return "\n".join(out)
    verdict = ("CLEANLY SEPARABLE" if bt["separable"]
               else "OVERLAPPING — no threshold separates them")
    out.append(f"  margin (min self - max non-self) = {bt['margin']:+.3f}  → {verdict}")
    out.append(f"  best threshold = {bt['threshold']:.3f}  "
               f"(accepts {bt['tpr']:.0%} of self, rejects {bt['tnr']:.0%} of non-self, "
               f"balanced acc {bt['balanced_acc']:.0%})")
    if bt["separable"]:
        lo = max(nonself_scores)
        hi = min(self_scores)
        out.append(f"  safe threshold band: ({lo:.3f}, {hi:.3f}] — "
                   f"midpoint {((lo + hi) / 2):.3f} is a robust pick")
    else:
        out.append("  Recommendation: per-turn ECAPA on these segments can't gate "
                   "reliably. Use longer audio windows / M-of-N smoothing, or improve "
                   "enrollment (longer, cleaner primary utterance).")
    return "\n".join(out)


def report(rows: list[ScoreRow], threshold_override: Optional[float], bins: int,
           label: Optional[str]) -> str:
    out: list[str] = []
    head = "=== speaker-verification score report ==="
    if label:
        head += f"  [label: {label}]"
    out.append(head)
    if not rows:
        out.append("No speaker scores found. (segment_scored is console-only; "
                   "barge_in[_rejected] only logged during --talkback sessions.)")
        return "\n".join(out)

    for source in ("turn_gate", "barge_in", "primary_match"):
        srows = [r for r in rows if r.source == source]
        if not srows:
            continue
        scores = [r.score for r in srows]
        thr = threshold_override
        if thr is None:
            thr = next((r.threshold for r in srows if r.threshold is not None),
                       DEFAULT_THRESHOLDS[source])
        st = summarize(scores)
        acc, rej = _accept_reject(srows, thr)
        out.append(f"\n--- source: {source}  (threshold {thr:.2f}) ---")
        out.append(f"  n={st['n']}  mean={st['mean']:.3f}  median={st['median']:.3f}  "
                   f"stdev={st['stdev']:.3f}")
        out.append(f"  range=[{st['min']:.3f}, {st['max']:.3f}]  "
                   f"p10={st['p10']:.3f}  p90={st['p90']:.3f}")
        out.append(f"  accept(>=thr)={acc}  reject(<thr)={rej}  "
                   f"accept_rate={acc/st['n']:.0%}")
        out.append(histogram(scores, bins=bins))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help=f"log files/globs (default: {DEFAULT_GLOB})")
    ap.add_argument("--session", help="only this session_id (substring match ok)")
    ap.add_argument("--source",
                    choices=["turn_gate", "barge_in", "primary_match", "all"],
                    default="all", help="which score source to include")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the accept/reject threshold")
    ap.add_argument("--label", help="tag the report (e.g. self / nonself)")
    ap.add_argument("--group-by", choices=["session"], default=None,
                    help="print a separate report per session")
    ap.add_argument("--self", dest="self_session",
                    help="session_id of the SELF run (enrolled speaker only)")
    ap.add_argument("--nonself", dest="nonself_session",
                    help="session_id of the NON-SELF run (other speaker's turns)")
    ap.add_argument("--bins", type=int, default=20, help="histogram bins (default 20)")
    args = ap.parse_args(argv)

    patterns = args.paths or [DEFAULT_GLOB]
    files = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not files:
        print(f"No log files matched: {patterns}")
        return 1

    rows = extract_scores(iter_records(files))
    if args.session:
        rows = [r for r in rows if args.session in r.session_id]
    if args.source != "all":
        rows = [r for r in rows if r.source == args.source]

    print(f"Scanned {len(files)} file(s); {len(rows)} score event(s).")

    # Labeled comparison mode: self session vs non-self session.
    if args.self_session or args.nonself_session:
        src = args.source if args.source != "all" else "turn_gate"
        cmp_rows = [r for r in rows if r.source == src]
        self_rows = [r for r in cmp_rows if args.self_session
                     and args.self_session in r.session_id]
        nonself_rows = [r for r in cmp_rows if args.nonself_session
                        and args.nonself_session in r.session_id]
        print(f"Comparison source: {src}  "
              f"(self n={len(self_rows)}, non-self n={len(nonself_rows)})")
        print(report_comparison(self_rows, nonself_rows, args.bins))
        return 0

    if args.group_by == "session":
        for sid in sorted({r.session_id for r in rows}):
            srows = [r for r in rows if r.session_id == sid]
            print(f"\n########## session {sid}  ({len(srows)} scores) ##########")
            print(report(srows, args.threshold, args.bins, args.label))
    else:
        print(report(rows, args.threshold, args.bins, args.label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
