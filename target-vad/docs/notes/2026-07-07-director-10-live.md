# Director-10 live validation verdict — array audio-out (2026-07-06/07)

Branch: `feat/director-10-array-audio-out`. TTS routed through the ReSpeaker's
3.5mm jack (powered speaker) so the XVF-3000's hardware AEC cancels our own
playback; capture moved to the array's processed ch0. Software AEC disabled by
config, kept in code.

## Merge gate — all four checks PASS

1. **Bug A structurally dead** (self-hearing). 2-minute story runs on both
   days: zero own-voice `NearFieldOnset` events during playback, `gain=1.0`
   throughout undisturbed stretches, zero self-replies. The talkback-era
   feedback loop cannot occur on this path.
2. **Routing pinned, fail-loud.** Startup resolves `output_device: "ReSpeaker"`
   against live `pw-dump` sinks and pins via `PIPEWIRE_NODE`
   (`✓ TTS output pinned: alsa_output.usb-SEEED_ReSpeaker...`); a missing sink
   or unreadable pw-dump exits 4 before the session starts.
3. **Capture is processed ch0.** 6-channel open with `use_channel: 0`
   (beamformed + hardware AEC + NS). The 1-channel downmix trap (raw capsules
   + ch5 playback reference folded into mono) is fixed and regression-tested.
4. **Barge-in works.** Live 2026-07-07 session 2, three consecutive cycles:
   interjection → Duck → served → Cut + new reply ("Tell me a story",
   "Stop one minute" → "Okay, I've paused", "Tell me a new story" → new story).

## Tuning decisions (measured, not guessed)

- `turn_gate.speaker_threshold` **stays 0.15**. On the ch0 substrate the owner's
  accumulated ECAPA windows scored 0.221/0.365 vs podcast 0.042/0.108; raising
  toward 0.25–0.30 would eat owner windows.
- `barge_in.proximity.rms_factor` **stays 0.5**. The planned raise is dead: the
  seed rms varies per session (0.085 → 0.202 across runs) and a podcast's rms
  (0.049–0.107) can overlap the owner's seed, so no factor separates them
  reliably. Loudness is a weak gate; identity + (later) direction are the real
  ones.
- **Loud-bystander suspension added** (`5870458`): while the safety net's miss
  streak is ≥ 2, new turns are accumulated (so one passing owner window
  unlocks) but never served, and the silence clock keeps running. This closes
  the 2026-07-07 run-1 hole where a podcast above the floor was served as user
  turns while the ECAPA verifier WARNed impotently (its eject rule required
  sub-floor rms). Unit-tested (6 tests); not yet exercised live — later runs
  never built a streak because the floor caught pure-podcast segments first.

## Known limitations carried forward

- **Overlapped speech: the louder source wins STT.** Live 2026-07-07 session 2:
  owner said "continue with the story" while the podcast played louder; the
  segment's ECAPA window correctly scored owner (0.474) so it was served, but
  STT transcribed the podcast ("The runway forms transformed into a major crime
  scene"). Identity gating cannot fix simultaneous overlap — this is source
  separation, i.e. Director-11 (DOA cone / beam selection) territory.
- **First-leak window:** a loud bystander's first turn can be served before the
  miss streak builds (~2 windows ≈ 6 s of audio).
- **Unlock cost:** after a suspension, the owner's first utterance is eaten
  (it is what unlocks); the second serves.
- **Head-down-at-phone = false ABSENT** (D07 gap): YuNet loses the face or
  SFace cosine drops to 0.18–0.42 (< 0.40), and after grace + talk-guard the
  session ends `owner_absent` — this ended 2026-07-07 session 1 mid-barge-in.
  Chosen workaround: face the kiosk / prop the phone up. Structural fix is a
  D07 follow-up, not D10.
- **Duck churn:** a podcast repeatedly triggers `NearFieldOnset` → Duck during
  SPEAKING, so playback ducks to 0.35 for much of a story while background
  audio plays. Cosmetic-to-annoying; D11's direction gate is the fix.

## Verdict

**MERGE.** The migration does what it was built for — the kiosk can no longer
hear itself — and every regression found during live validation was
root-caused and fixed on the branch (PipeWire card-reservation crash, downmix
capture, presence/floor DIAG gaps, loud-bystander suspension). Suite:
726 passed / 2 skipped.
