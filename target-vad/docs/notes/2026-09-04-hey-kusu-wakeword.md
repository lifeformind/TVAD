# "Hey Kusu" custom wake word (2026-09-04)

The kiosk's wake phrase is now **"Hey Kusu"** (espeak: `h'eI k'u:su:`) — after
the Kusu Island legend: the giant turtle that turned itself into an island to
save shipwrecked sailors. Chosen for Singapore cultural resonance, a
helpful-spirit connotation fitting a service kiosk, no trademark exposure
(unlike mascots Merli/Singa), 3 crisp syllables, and near-zero occurrence in
ambient speech.

Shipped artifacts:
- `models/wake/hey_kusu.onnx` — the trained openWakeWord model (git-tracked,
  ~215KB, weights consolidated internal, opset 18).
- `config.yaml`: `kiosk.wake_phrase: "models/wake/hey_kusu.onnx"`,
  `wake_threshold: 0.4`.
- `WakeWordDetector` matches prediction keys by `Path(model_name).stem`, so
  name-or-path values both work.

## Training environment (GB10-local, outside the repo)

`~/.local/opt/oww-train` (override via `OWW_TRAIN_DIR`); driver:
`bench/train_wakeword.sh [config.yml]` (default `hey_kusu.yml`). ONNX-only —
we never pass `--convert_to_tflite` (the kiosk detector runs onnxruntime).

Layout:
- `venv/` — `--system-site-packages` venv (reuses the system CUDA torch).
- `openWakeWord/` — github.com/dscripka/openWakeWord clone (training code is
  not in the pip package). Feature-extractor models (melspectrogram/embedding
  onnx+tflite, v0.5.1 release) downloaded into
  `openwakeword/resources/models/`.
- `piper-sample-generator/` — the **dscripka fork** (flat layout with
  `generate_samples.py`, which train.py imports; the rhasspy repo repackaged
  it). `en_US-libritts_r-medium.pt` checkpoint from the rhasspy v2.0.0
  release, symlinked over the default `en-us-libritts-high.pt` name because
  train.py does not pass `model=`.
- `hey_kusu.yml` — training config: 30k train / 2k val positives, 50k steps,
  dnn/layer_size 32, ACAV100M precomputed negatives, target 0.2 FP/hr.
- Data: `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` (17.3GB) +
  `validation_set_features.npy` (HF davidscripka/openwakeword_features),
  `mit_rirs/` (270 RIRs), `audioset_16k/` (3,400 clips streamed from
  agkphysics/AudioSet parquet — the notebook's tar layout is gone; FMA was
  dropped because rudraml/fma needs `trust_remote_code`, AudioSet's
  balanced split is music-heavy anyway).
- `run_train.py` — runner that applies `_oww_ta_patch` (see below) then
  executes train.py as `__main__`.
- `validate_kusu.py` — offline threshold sweep (see Validation).
- `download_aug_data.py`, `smoke_test.py` — setup helpers.

### Environment patches (why they exist)

torch 2.12 / new-stack friction, all contained to the training env:
- `torch.load(..., weights_only=False)` in `generate_samples.py` and in the
  venv's `dp/model/model.py` (piper + deep-phonemizer checkpoints are full
  pickles; torch≥2.6 flipped the default).
- `_oww_ta_patch.py` (venv site-packages): system torchaudio 2.11 delegates
  `load`/`info` to torchcodec, which needs FFmpeg libs the box lacks — the
  patch rebinds both to soundfile (every training clip is a plain 16k WAV).
  Applied via `run_train.py` import, NOT sitecustomize (Ubuntu's
  `/usr/lib/python3.12/sitecustomize.py` wins that name).
- venv pins: `huggingface_hub>=1.5` (system transformers 5.9 needs it;
  datasets was only used for downloads), `torch-audiomentations` upgraded
  (0.11 called removed `torchaudio.set_audio_backend`), `scipy<1.17`
  (`acoustics` imports removed `sph_harm`), `onnxscript` (torch 2.12 ONNX
  export needs it; the opset-13→18 conversion warning is non-fatal — the
  model exports at opset 18, fine for onnxruntime).
- torch 2.12 exports external weights (`.onnx.data`); the shipped model was
  consolidated to a single file via `onnx.load()`/`onnx.save()`.

### Gotcha: partial feature files

`compute_features_from_generator` preallocates the full memmap up front, so
a killed run leaves a **full-size but partly-zero** `.npy`, and the augment
stage's existence check will then skip augmentation entirely (this bit us:
run 3 trained against garbage until the missing test file crashed it).
Before resuming a killed run, verify with zero-row counts and delete any
partial file.

## Validation (2026-09-04, validate_kusu.py)

2,000 held-out synthetic positives (hard: RIR+noise augmented), 2,000
synthetic negatives, plus 19 real room recordings (0.5 min) from
`debug_audio/` (the owner speaking near the kiosk — the FA material that
matters):

| thr  | accept% | syn-FA% | real-audio FA |
|------|---------|---------|----------------|
| 0.20 | 71.8    | 0.25    | 0 |
| 0.30 | 66.1    | 0.15    | 0 |
| 0.40 | 61.5    | 0.10    | 0 |
| 0.50 | 56.6    | 0.05    | 0 |

Real-audio peaks were all ≈0.001 — no near-misses. Kiosk-env smoke test:
14/20 synthetic holdouts detected at 0.4 through the actual
`WakeWordDetector`. Auto-training final val: accuracy 0.65, recall 0.31,
0.0 FP/hr (on the deliberately hard augmented set — live close-mic speech
scores far higher; the prebuilt models bench similarly on this eval).

Threshold 0.4 shipped; sweep live from the tune console if wake feels deaf
(drop toward 0.3) or trigger-happy (raise toward 0.5). If synthetic-trained
quality disappoints live, the fallbacks are: more `n_samples` (100k+),
`augmentation_rounds: 2`, adding recorded real positives, or Porcupine
(closed-source, needs access key, replaces the detector).
