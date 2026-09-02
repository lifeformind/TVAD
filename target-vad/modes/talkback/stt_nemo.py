"""Parakeet-TDT STT backend via NeMo (spec Appendix B: ~6.05% vs ~8.6% WER).

Satisfies the StreamingStt contract (modes/talkback/stt.py): eager
_ensure_model() warm-up, whole-segment async transcribe_segment ->
TranscriptResult. NeMo import stays inside methods so the module imports
on machines without nemo (conftest does not stub it).

Confidence: nvidia/parakeet-tdt-0.6b-v2 ships with word_confidence OFF by
default (hyp.word_confidence exists on every Hypothesis but is empty). Task
9's spike (docs: .superpowers/sdd/2026-09-02-quick-wins-upgrades/task-9-report.md
section 5) proved the one-time decoding-strategy change below turns it on.

Confidence METHOD matters as much as turning it on: the spike's default
method_cfg (name="entropy") measured mean_word_prob=0.24 on an exact-match
transcript of self.wav — well under the reducer's conf_floor (0.5), which
would have silently RESTORE'd every genuine turn as non-content. Switching
method_cfg to name="max_prob" (softmax-probability-scale confidence, the
scale whisper's word probabilities were tuned against) measured
mean=0.989 (self.wav) / mean=0.957 (other.wav) on the same exact transcripts
-- see task-10-report.md section "Confidence method fix" for the full
before/after. max_prob is baked into _ensure_model() below.
"""
import asyncio

import numpy as np

from modes.director.transcript import TranscriptResult


class NemoStt:
    def __init__(self, model: str = "nvidia/parakeet-tdt-0.6b-v2", device: str = "cuda"):
        self._model_name = model
        self._device = device
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import nemo.collections.asr as nemo_asr
        from omegaconf import open_dict
        from nemo.collections.asr.parts.utils.asr_confidence_utils import (
            ConfidenceConfig, ConfidenceMethodConfig)

        model = nemo_asr.models.ASRModel.from_pretrained(self._model_name)
        model = model.to(self._device).eval()

        # Enable per-word confidence (spike-verified invocation, task-9-report.md
        # section 5). decoding_cfg is a struct DictConfig; confidence_cfg is not
        # a pre-existing key on the TDT/RNNT decoding schema, so a plain
        # attribute-set raises "Key 'confidence_cfg' is not in struct" without
        # open_dict() disabling struct-mode just long enough to add the key.
        #
        # method_cfg name="max_prob" (NOT the default "entropy"): entropy-scale
        # confidence measured mean_word_prob=0.24 on an exact-match transcript
        # (self.wav) -- below conf_floor (0.5), so every genuine turn would be
        # RESTORE'd as non-content. max_prob is softmax-probability-scale, the
        # same scale whisper's word probabilities (and conf_floor) were tuned
        # against; measured mean=0.989 (self.wav) / 0.957 (other.wav) on the
        # same exact transcripts (see task-10-report.md).
        decoding_cfg = model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.confidence_cfg = ConfidenceConfig(
                preserve_word_confidence=True,
                preserve_token_confidence=True,
                preserve_frame_confidence=False,
                method_cfg=ConfidenceMethodConfig(name="max_prob"),
            )
        model.change_decoding_strategy(decoding_cfg)

        self._model = model

    async def transcribe_segment(self, audio: np.ndarray) -> TranscriptResult:
        self._ensure_model()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> TranscriptResult:
        hyps = self._model.transcribe(
            audio=[audio], return_hypotheses=True, verbose=False)
        if not hyps:
            return TranscriptResult(text="", mean_word_prob=0.0)
        hyp = hyps[0]
        # Some NeMo versions nest single-output batches in a list-of-lists
        # (bench/stt_backend_probe.py._run_transcribe carries the same defense).
        if isinstance(hyp, list) and hyp:
            hyp = hyp[0]
        text = (hyp.text or "").strip()
        conf = getattr(hyp, "word_confidence", None)
        if not text:
            prob = 0.0
        elif conf:
            prob = float(np.clip(np.mean(conf), 0.0, 1.0))
        else:
            prob = 1.0   # no confidence stream -> don't false-trip the conf_floor gate
        return TranscriptResult(text=text, mean_word_prob=prob)
