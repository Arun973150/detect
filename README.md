# PikSign

Two-branch AI-generated image detector targeting **Nano Banana** (Gemini image
gen) and **GPT-4o** images, robust to social-media laundering (recompression,
resizing, metadata stripping). Built on the *Task–Model Alignment* principle
(AlignGemini, arXiv 2512.06746): each model is trained only on the subtask its
inductive bias is suited for.

```
input image ─┬─ C. provenance (C2PA/EXIF rules, free instant catches)
             ├─ A. pixel experts (DINOv2+LoRA, native-res crops, top3 agg)
             │     a1_sd21        SD2.1-VAE reconstruction pairs   (generalization)
             │     a2_flux        FLUX-VAE reconstruction pairs    (generalization)
             │     a3_nanobanana  Pico-Banana-400K edit pairs      (target #1)
             │     a4_gpt4o       ShareGPT-4o-Image edit pairs     (target #2)
             └─ B. semantic VLM (Qwen2.5-VL-7B + DPO LoRA, pixel-scrambled corpus)
                        │
             fusion: per-branch thresholds under a global 5% FPR budget, OR rule
```

Design rules enforced throughout the code (do not undo them):

- **Never resize** an image to fit a network — pixel fingerprints live at
  native resolution; we crop instead ([datasets.py](src/piksign/datasets.py)).
- **One normalization funnel** (JPEG q95 4:4:4) for every training image of
  both classes, so format can never become the label
  ([download/\_\_init\_\_.py](src/piksign/download/__init__.py)).
- **Shared laundering augmentation** across classes during expert training
  ([launder.py](src/piksign/launder.py)) — compression can't become the label either.
- **Aligned pairs**: each fake is trained against the real it was derived
  from, resized real→fake dimensions, so only the generator fingerprint separates them.
- **Pixel-scrambled semantic corpus**: the VLM branch never sees usable
  low-level cues, so it can only learn semantics.
- Fresh eval fakes are **generated at eval time via the live APIs** — never
  trained on, different prompt distribution: a true out-of-distribution test.

## Setup

```bash
pip install -e ".[dpo,gen]"        # training + eval-generation
# vLLM only for the DPO labeling phase (it pins its own torch):
pip install -e ".[label]"
```

Environment (defaults in parentheses):

```bash
export PIKSIGN_DATA=/workspace/data          # (./data)
export PIKSIGN_CKPT=/workspace/checkpoints   # (./checkpoints)
export HF_HOME=/workspace/hf                 # keep the model cache on the volume
export GEMINI_API_KEY=...                    # only for gen_eval_api
export OPENAI_API_KEY=...                    # only for gen_eval_api
```

## RunPod runbook (A100 SXM 80GB, 50GB container + 250GB network volume)

```bash
bash scripts/runpod_setup.sh       # env + install
bash scripts/00_download_all.sh    # CPU-bound; a cheap CPU pod on the same volume works
bash scripts/01_recon.sh           # GPU ~2h : SD2.1 + FLUX VAE reconstructions
bash scripts/02_train_experts.sh   # GPU ~6-8h: A1-A4
bash scripts/03_train_vlm.sh       # GPU ~10h : 72B labeling + 7B DPO
# generate fresh eval fakes (~$30 total, needs API keys):
python -m piksign.download.gen_eval_api --provider gemini --n 400 --style mixed --yes
python -m piksign.download.gen_eval_api --provider openai --n 400 --style mixed --yes
bash scripts/04_eval.sh            # GPU ~2h : launder twins, score, calibrate, report
```

Stop the pod between phases — everything lives on the network volume.

## Budget mode (~$24 of RunPod credits)

The A100 is only genuinely needed for the VLM phase. Allocation that fits $24:

| Phase | Pod | Est. |
|---|---|---|
| Network volume **150GB**, ~2 weeks, delete when done | — | ~$5 |
| `00_download_all.sh` | CPU pod ($0.05–0.10/hr, ~5h) | ~$0.5 |
| `01_recon.sh` + `02_train_experts.sh` + expert eval | RTX 4090 24GB (~$0.35–0.69/hr, ~9h) | ~$4–6 |
| `03_train_vlm.sh` with `LABEL_MODEL=Qwen/Qwen2.5-VL-32B-Instruct-AWQ` | A100 80GB ($1.50/hr, ~6h) | ~$9 |
| VLM eval scoring | tail of the A100 session or 4090 | ~$1 |

**Total ≈ $17–19, leaving ~$5 buffer.** Rules that keep it there: STOP the pod
the moment a phase ends (billing is per-millisecond); never re-download (the
volume persists); run `--delete-zip` on COCO; and if credits run low, ship v1
without the DPO branch — the harness/CLI fall back to the base 7B VLM
automatically, which loses only a few points in the wild (per the paper), and
the DPO run can be added later with `03_train_vlm.sh` unchanged.

Fresh eval fakes are API costs, not GPU credits: Gemini's free tier covers
Nano Banana generations (spread over a few days of daily quota), and ~$5 of
OpenAI credit covers ~120 `gpt-image-1` images.

## Using the detector

```bash
piksign check photo.jpg                    # pixel experts + provenance
piksign check folder/ --vlm --rationale    # + semantic branch with explanation
piksign check img.png --json
```

Output per image: fused verdict, which branches fired, per-branch scores vs
calibrated thresholds, provenance metadata, optional VLM rationale.

## Evaluation model

Eval tree: `data/eval/<variant>/<source>/{real,fake}/`. Variants: `clean`,
`whatsapp` (1280px + JPEG70), `double_jpeg`, `screenshot`, `double_resize`
(the hardest). Sources include `nanobanana_fresh` / `gpt4o_fresh` (live-API
generations) and `reals_coco` / `reals_phone`.

Headline metric: balanced accuracy per (source, variant) at the calibrated
5%-global-FPR operating point; per-branch AUC / TPR@5%FPR as diagnostics.
`data/eval/scores.jsonl` is append-only and resumable; the report dedupes.

Re-run `04_eval.sh` whenever Google/OpenAI ship new model versions — it
doubles as the drift monitor. If a target expert (a3/a4) decays, re-collect
pairs and retrain just that adapter (~2 GPU-hours).

## Data sources & licenses

| Set | Source | License / note |
|---|---|---|
| Nano Banana pairs | [Pico-Banana-400K](https://github.com/apple/pico-banana-400k) (Apple CDN) + OpenImages S3 | **CC BY-NC-ND 4.0 — research/non-commercial only** |
| GPT-4o pairs | [ShareGPT-4o-Image](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image) | check hub card; audit whether edit inputs are genuine photos |
| Semantic fakes | [Echo-4o-Image](https://huggingface.co/datasets/Yejy53/Echo-4o-Image) surreal subset | open GPT-4o generations |
| Reals | COCO (train/val 2017), OpenImages | CC-BY family |
| VAEs | SD 2.1 vae, FLUX.1-schnell vae | schnell is Apache-2.0 (same AE as dev, ungated) |

The piksign Drive copies (`pos.zip`/`neg.zip`) are heavily recompressed
(~55KB/img) and are only a fallback; training uses re-downloaded originals.

## Known limits (stated honestly)

- Heavily laundered **subtle** edits are the field's frontier: expect degraded
  (not solved) accuracy there; `double_resize` numbers show the floor.
- Splice-style local AI edits (generative fill) only fire if a crop lands on
  the region — the top3/max crop aggregation is the mitigation, not a fix.
- Provenance byte-scan is heuristic, not cryptographic verification (install
  `c2patool` for real manifest parsing).
- a3/a4 decay when target models update; a1/a2 are the stable backstop.
- ShareGPT-4o-Image "real" inputs are of mixed provenance — if `a4` looks too
  good on val but bad on `gpt4o_fresh`, retrain it with `--real-dir` pointed
  at COCO reals instead.
