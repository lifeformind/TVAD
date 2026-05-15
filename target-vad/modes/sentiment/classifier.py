"""SentimentClassifier — thin wrapper around two transformers.pipeline instances.

Both pipelines load lazily on first classify_batch() call: model construction
downloads weights on cold cache (~500 MB polarity + ~330 MB emotion), so
deferring lets unit tests construct the classifier without triggering network
calls. A public load() method is provided for eager loading so download failures
surface early as the orchestration's EXIT_MODEL_OR_IO rather than being masked
by per-batch exception handlers.

Both models are passed through to transformers.pipeline() as-is. The two
preset model identifiers (polarity_model, emotion_model) are strings — any
HuggingFace model id, local model directory, or future custom model with the
right output shape works via config alone.

Label validation: pipeline output labels are checked against canonical sets.
Non-canonical labels raise ValueError with a message identifying the offending
model. This prevents silent schema drift when a config-swapped model emits
unexpected labels (e.g., an engagement-oriented classifier configured in the
polarity slot).
"""

from typing import Dict, List

from transformers import pipeline as _hf_pipeline

# Local alias so tests can monkeypatch via `modes.sentiment.classifier.pipeline`.
pipeline = _hf_pipeline


_POLARITY_LABELS = ("positive", "neutral", "negative")
_EMOTION_LABELS = ("joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral")


class SentimentClassifier:
    """Classifies texts with a polarity model + an emotion model in one batch call."""

    def __init__(self, polarity_model: str, emotion_model: str, device: str):
        self._polarity_model = polarity_model
        self._emotion_model = emotion_model
        self._device = device
        self._polarity_pipe = None
        self._emotion_pipe = None

    def load(self) -> None:
        """Eagerly load both pipelines. Use to surface download/load errors before the loop."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._polarity_pipe is None:
            self._polarity_pipe = pipeline(
                "text-classification",
                model=self._polarity_model,
                top_k=None,
                device=self._device,
            )
        if self._emotion_pipe is None:
            self._emotion_pipe = pipeline(
                "text-classification",
                model=self._emotion_model,
                top_k=None,
                device=self._device,
            )

    def classify_batch(self, texts: List[str]) -> List[Dict]:
        """Classify a batch of texts. Returns one nested {polarity, emotion} dict per text.

        Args:
            texts: list of non-empty strings. Callers MUST filter out None / "" texts
                upstream — this method assumes all inputs are real text.

        Returns:
            List of dicts, one per input text in order:
            {
                "polarity": {"label": "neutral", "score": 0.72, "scores": {"positive": ..., ...}},
                "emotion":  {"label": "surprise", "score": 0.51, "scores": {"joy": ..., ...}},
            }
        """
        if not texts:
            return []

        self._ensure_loaded()
        polarity_raw = self._polarity_pipe(texts)
        emotion_raw = self._emotion_pipe(texts)

        results: List[Dict] = []
        for pol_scores_list, emo_scores_list in zip(polarity_raw, emotion_raw):
            polarity = _build_block(pol_scores_list, _POLARITY_LABELS, model_kind="polarity")
            emotion = _build_block(emo_scores_list, _EMOTION_LABELS, model_kind="emotion")
            results.append({"polarity": polarity, "emotion": emotion})
        return results


def _build_block(scores_list: List[Dict], canonical_labels: tuple, model_kind: str) -> Dict:
    """Convert pipeline output (List[{label, score}]) to the spec's nested block shape.

    Validates that every label is in `canonical_labels`. Raises ValueError on any
    label outside the canonical set, naming the offending model_kind.
    """
    scores: Dict[str, float] = {}
    for entry in scores_list:
        label = entry["label"]
        if label not in canonical_labels:
            raise ValueError(
                f"{model_kind} model emitted label {label!r} which is not in the canonical "
                f"set {canonical_labels!r}. Check the model's id2label mapping or use a "
                f"different model in config.sentiment.{model_kind}_model."
            )
        scores[label] = float(entry["score"])

    # argmax label
    best_label = max(scores, key=scores.get)
    return {
        "label": best_label,
        "score": scores[best_label],
        "scores": scores,
    }
