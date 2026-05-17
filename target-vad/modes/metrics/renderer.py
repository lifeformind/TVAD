# -*- coding: utf-8 -*-
"""Render the contribution_metrics block as a human-readable Markdown report.

Pure function: same metrics + session metadata -> identical Markdown. No LLM,
no I/O. The caller writes the returned string to disk in UTF-8.

Section omission rules (per spec):
  - `## Notable moments` omitted when highlights list is empty
  - `## Who follows whom` omitted when there is only one speaker
  - `## Activity over time` omitted when timeline has only one bucket
  - `## Prosody (per speaker)` omitted when prosody_baselines is None or empty
"""

import os
from typing import Dict, List


_BLOCK_LADDER = [(0.001, " "), (0.25, "░"), (0.5, "▒"), (0.75, "▓"), (1.01, "█")]


def _fmt_seconds_compact(s: float) -> str:
    if s >= 60:
        mins = int(s // 60)
        rem = s - mins * 60
        if rem < 0.05:
            return f"{mins} min"
        rem_int = int(rem)
        if rem == rem_int:
            return f"{mins} min {rem_int} s"
        return f"{mins} min {rem:.1f} s"
    return f"{s:.1f} s"


def _fmt_mmss(s: float) -> str:
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m:02d}:{sec:02d}"


def _fmt_pct(p: float) -> str:
    # Render whole-number percentages with no decimal, fractional with one.
    if p == int(p):
        return f"{int(p)}%"
    return f"{p:.1f}%"


def _fmt_quote(q: str) -> str:
    return f'*"{q}"*' if q else ""


def render_markdown(metrics: Dict, session_meta: Dict, prosody_baselines: Dict | None = None) -> str:
    lines: List[str] = []

    audio = session_meta.get("audio_file", "(unknown source)")
    audio_name = os.path.basename(audio) if audio else "(unknown source)"
    lines.append(f"# Session Metrics — {audio_name}")
    lines.append("")

    sess = metrics["session"]
    duration_pretty = _fmt_seconds_compact(sess["duration_s"])
    lines.append(
        f"**Duration:** {sess['duration_s']} s ({duration_pretty}) · "
        f"**Speech:** {sess['speech_duration_s']} s · "
        f"**Silence:** {sess['silence_duration_s']} s"
    )
    catchall = 1 if sess.get("unknown_segments", 0) > 0 else 0
    recurring_unknown = sess.get("recurring_unknown_speakers", 0)
    enrolled = sess.get("identified_speakers", 0)
    speakers_total = enrolled + recurring_unknown + catchall
    lines.append(
        f"**Speakers:** {speakers_total} "
        f"({enrolled} enrolled, {recurring_unknown} recurring unknown, {catchall} catchall) · "
        f"**Words:** {sess['total_words']} · **Segments:** {sess['total_segments']}"
    )
    lines.append(f"**Analyzed:** {session_meta.get('analyzed_at', '')}")
    lines.append("")

    highlights = metrics.get("highlights") or []
    if highlights:
        lines.append("## Notable moments")
        speaker_names = {sp["speaker_id"]: sp["speaker"] for sp in metrics["speakers"]}
        for h in highlights:
            lines.append(_render_highlight(h, speaker_names))
        lines.append("")

    # Participation table.
    lines.append("## Participation")
    lines.append("")
    lines.append("| Speaker   | Talk  | %     | Segs | Words | WPM   | Mean seg | Max seg |")
    lines.append("|-----------|------:|------:|-----:|------:|------:|---------:|--------:|")
    for sp in metrics["speakers"]:
        p = sp["participation"]
        wpm = f"{p['words_per_minute']:.1f}" if p["words_per_minute"] is not None else "—"
        lines.append(
            f"| {sp['speaker']} | {p['talk_seconds']}s | {_fmt_pct(p['talk_percent'])} | "
            f"{p['segment_count']} | {p['word_count']} | {wpm} | "
            f"{p['mean_segment_seconds']} s | {p['max_segment_seconds']} s |"
        )
    lines.append("")

    # Polarity table.
    lines.append("## Sentiment — polarity (per speaker)")
    lines.append("")
    lines.append("| Speaker   | Positive | Neutral | Negative | Mean conf. |")
    lines.append("|-----------|---------:|--------:|---------:|-----------:|")
    for sp in metrics["speakers"]:
        pol = sp["sentiment"]["polarity"]["percent"]
        conf = sp["sentiment"]["polarity"]["mean_top_confidence"]
        conf_s = f"{conf:.2f}" if conf is not None else "—"
        lines.append(
            f"| {sp['speaker']} | {_fmt_pct(pol['positive'])} | "
            f"{_fmt_pct(pol['neutral'])} | {_fmt_pct(pol['negative'])} | {conf_s} |"
        )
    lines.append("")

    # Emotion table.
    lines.append("## Sentiment — emotion (per speaker)")
    lines.append("")
    lines.append("| Speaker   | Joy | Neutral | Surprise | Disgust* | Anger | Fear | Sadness | Mean conf. |")
    lines.append("|-----------|----:|--------:|---------:|---------:|------:|-----:|--------:|-----------:|")
    for sp in metrics["speakers"]:
        emo = sp["sentiment"]["emotion"]["percent"]
        conf = sp["sentiment"]["emotion"]["mean_top_confidence"]
        conf_s = f"{conf:.2f}" if conf is not None else "—"
        lines.append(
            f"| {sp['speaker']} | {_fmt_pct(emo['joy'])} | {_fmt_pct(emo['neutral'])} | "
            f"{_fmt_pct(emo['surprise'])} | {_fmt_pct(emo['disgust'])} | "
            f"{_fmt_pct(emo['anger'])} | {_fmt_pct(emo['fear'])} | {_fmt_pct(emo['sadness'])} | {conf_s} |"
        )
    lines.append("")
    lines.append(
        '\\* "Disgust" from the emotion model tends to fire on polite-disagreement phrasing — '
        'read as "registered disagreement" rather than visceral disgust.'
    )
    lines.append("")

    # Turn-taking table.
    lines.append("## Turn-taking")
    lines.append("")
    lines.append("| Speaker   | Turns | Mean gap before | Interruptions |")
    lines.append("|-----------|------:|----------------:|--------------:|")
    for sp in metrics["speakers"]:
        tt = sp["turn_taking"]
        gap = f"{tt['mean_gap_before_seconds']} s" if tt["mean_gap_before_seconds"] is not None else "—"
        lines.append(
            f"| {sp['speaker']} | {tt['turn_count']} | {gap} | {tt['interruption_count']} |"
        )
    lines.append("")

    # Prosody (per speaker) — only emitted when prosody data is present.
    if prosody_baselines:
        lines.append("")
        lines.append("## Prosody (per speaker)")
        lines.append("")
        lines.append("| Speaker   | Pitch median | Pitch IQR | Energy median | Energy IQR | Segments |")
        lines.append("|-----------|-------------:|----------:|--------------:|-----------:|---------:|")
        for sp in metrics["speakers"]:
            sid = sp["speaker_id"]
            name = sp["speaker"]
            b = prosody_baselines.get(sid)
            if b is None:
                continue  # speaker has no prosody data — skip the row
            def _fmt(v, unit):
                return f"{v} {unit}" if v is not None else "—"
            row = (
                f"| {name} "
                f"| {_fmt(b.get('pitch_hz_median'), 'Hz')} "
                f"| {_fmt(b.get('pitch_hz_iqr'), 'Hz')} "
                f"| {_fmt(b.get('energy_db_median'), 'dB')} "
                f"| {_fmt(b.get('energy_db_iqr'), 'dB')} "
                f"| {b.get('segment_count', 0)} |"
            )
            lines.append(row)
        lines.append("")

    # Who follows whom — only when >= 2 speakers.
    if len(metrics["speakers"]) >= 2:
        lines.append("## Who follows whom")
        lines.append("")
        lines.append("Rows = previous speaker, columns = next speaker. Cell = transition count.")
        lines.append("")
        speaker_ids = [sp["speaker_id"] for sp in metrics["speakers"]]
        speaker_names = {sp["speaker_id"]: sp["speaker"] for sp in metrics["speakers"]}
        header_cells = [f"→ {speaker_names[sid]}" for sid in speaker_ids]
        lines.append("|              | " + " | ".join(header_cells) + " |")
        sep_cells = ["----:" for _ in speaker_ids]
        lines.append("|--------------|" + "|".join(f" {c} " for c in sep_cells) + "|")
        pairwise = metrics.get("pairwise_followers", {})
        for sid in speaker_ids:
            row_cells = []
            for other in speaker_ids:
                if other == sid:
                    row_cells.append("—")
                else:
                    row_cells.append(str(pairwise.get(sid, {}).get(other, 0)))
            lines.append(f"| {speaker_names[sid]} → | " + " | ".join(row_cells) + " |")
        lines.append("")

    # Activity over time — only when >= 2 buckets.
    if len(metrics.get("timeline", [])) >= 2:
        lines.append("## Activity over time (5-min windows)")
        lines.append("")
        bucket_seconds = metrics["timeline"][0]["bucket_end_s"] - metrics["timeline"][0]["bucket_start_s"]
        # Header row of bucket-start timestamps.
        speaker_names = {sp["speaker_id"]: sp["speaker"] for sp in metrics["speakers"]}
        lines.append("```")
        timestamps = "        " + "  ".join(_fmt_mmss(b["bucket_start_s"]) for b in metrics["timeline"])
        lines.append(timestamps)
        for sp in metrics["speakers"]:
            cells = []
            for b in metrics["timeline"]:
                talk = b["per_speaker_talk_s"].get(sp["speaker_id"], 0.0)
                ratio = (talk / bucket_seconds) if bucket_seconds else 0
                cells.append(_block_for_ratio(ratio))
            label = sp["speaker"][:6].ljust(6)
            lines.append(f"{label}  " + "  ".join(cells))
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append(
        "_Caveat: catchall 'unknown' segments may still bundle multiple brief or "
        "one-off speakers; substantive recurring voices are surfaced as their own "
        "`SPEAKER_NN` rows above._"
    )
    return "\n".join(lines)


def _render_highlight(h: Dict, speaker_names: Dict[str, str]) -> str:
    kind = h["kind"]
    if kind == "longest_segment":
        name = speaker_names.get(h["speaker_id"], h["speaker_id"])
        return f"- **Longest contribution** — {name}, {h['value_s']} s at {_fmt_mmss(h['start'])}: {_fmt_quote(h['quote'])}"
    if kind == "most_positive":
        name = speaker_names.get(h["speaker_id"], h["speaker_id"])
        return f"- **Most positive** — {name} at {_fmt_mmss(h['start'])} (score {h['polarity_score']:.2f}): {_fmt_quote(h['quote'])}"
    if kind == "most_negative":
        name = speaker_names.get(h["speaker_id"], h["speaker_id"])
        return f"- **Most negative** — {name} at {_fmt_mmss(h['start'])} (score {h['polarity_score']:.2f}): {_fmt_quote(h['quote'])}"
    if kind == "high_disgust_window":
        name = speaker_names.get(h["speaker_id"], h["speaker_id"])
        return (f"- **High disgust window** — {_fmt_mmss(h['bucket_start_s'])}–"
                f"{_fmt_mmss(h['bucket_end_s'])}, mostly {name} ({h['count']} segments)")
    if kind == "quietest_window":
        return (f"- **Quietest window** — {_fmt_mmss(h['bucket_start_s'])}–"
                f"{_fmt_mmss(h['bucket_end_s'])} ({h['total_talk_s']} s talk)")
    if kind == "busiest_window":
        return (f"- **Busiest window** — {_fmt_mmss(h['bucket_start_s'])}–"
                f"{_fmt_mmss(h['bucket_end_s'])} ({h['total_talk_s']} s talk)")
    if kind == "solo_dominator":
        name = speaker_names.get(h["speaker_id"], h["speaker_id"])
        return (f"- **Solo dominator window** — {_fmt_mmss(h['bucket_start_s'])}–"
                f"{_fmt_mmss(h['bucket_end_s'])}, {name} held {h['talk_s']} of {h['total_talk_s']} s")
    return f"- **{kind}** — {h}"


def _block_for_ratio(r: float) -> str:
    for threshold, glyph in _BLOCK_LADDER:
        if r < threshold:
            return glyph
    return "█"
