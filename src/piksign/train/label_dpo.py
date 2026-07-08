"""Generate DPO preference pairs for the semantic branch, locally via vLLM.

Protocol (per the AlignGemini paper): for each image with a KNOWN label, a
large VLM (Qwen2.5-VL-72B-AWQ fits on one A100-80GB) is prompted twice with a
prefilled verdict sentence - once correct, once wrong - and asked to justify
from semantic cues only. The correct-verdict response becomes `chosen`, the
wrong-verdict response `rejected`. No human annotation, no VLM-as-judge.

Images are pixel-scrambled (heavy launder) BEFORE labeling so both labeler and
trainee only ever see the semantics-only version of the corpus. The scrambled
copies are saved and reused by train_dpo.py.

Input:   data/processed/semantic/{real,fake}/*.jpg
Output:  data/processed/semantic_scrambled/{real,fake}/*.jpg
         data/processed/dpo_pairs.jsonl

    python -m piksign.train.label_dpo --model Qwen/Qwen2.5-VL-72B-Instruct-AWQ
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ..launder import SEMANTIC_SCRAMBLE, random_launder, resize_to_max_side
from ..models.vlm import PROMPT, VERDICT_FAKE, VERDICT_REAL
from ..paths import ensure, list_images, processed_dir

JUSTIFY_SUFFIX = " "  # model continues after the verdict sentence


def scramble_corpus(seed: int) -> list[tuple[Path, int]]:
    """Heavy-launder every semantic image once; deterministic, resumable."""
    out_items: list[tuple[Path, int]] = []
    for cls, label in (("real", 0), ("fake", 1)):
        src_dir = processed_dir("semantic", cls)
        dst_dir = ensure(processed_dir("semantic_scrambled", cls))
        for p in tqdm(list_images(src_dir), desc=f"scramble {cls}"):
            dst = dst_dir / p.name
            if not dst.exists():
                rng = random.Random(f"{seed}:{cls}:{p.name}")
                with Image.open(p) as im:
                    img = random_launder(im.convert("RGB"), rng, SEMANTIC_SCRAMBLE)
                img.save(dst, format="JPEG", quality=95, subsampling=0)
            out_items.append((dst, label))
    return out_items


def build_prompt_text(processor) -> str:
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
    }]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct-AWQ")
    ap.add_argument("--out", type=Path, default=None, help="default data/processed/dpo_pairs.jsonl")
    ap.add_argument("--max-side", type=int, default=896)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = args.out or processed_dir("dpo_pairs.jsonl")
    items = scramble_corpus(args.seed)
    print(f"{len(items)} scrambled images")

    done: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["image"])
                except Exception:  # noqa: BLE001
                    pass
        print(f"resuming: {len(done)} already labeled")
    todo = [(p, l) for p, l in items if str(p) not in done]

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(args.model)
    prompt_text = build_prompt_text(processor)
    llm = LLM(
        model=args.model,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_mem,
        limit_mm_per_prompt={"image": 1},
    )
    params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=args.max_tokens)

    data_root_str = str(processed_dir())
    with open(out_path, "a", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(todo), args.batch), desc="label"):
            chunk = todo[i: i + args.batch]
            requests, meta = [], []
            for p, label in chunk:
                try:
                    with Image.open(p) as im:
                        img = resize_to_max_side(im.convert("RGB"), args.max_side)
                except Exception as e:  # noqa: BLE001
                    print(f"[skip {p}] {e}")
                    continue
                correct = VERDICT_FAKE if label == 1 else VERDICT_REAL
                wrong = VERDICT_REAL if label == 1 else VERDICT_FAKE
                for verdict in (correct, wrong):
                    requests.append({
                        "prompt": prompt_text + verdict + JUSTIFY_SUFFIX,
                        "multi_modal_data": {"image": img},
                    })
                meta.append((p, label, correct, wrong))
            if not requests:
                continue
            outputs = llm.generate(requests, params)
            for j, (p, label, correct, wrong) in enumerate(meta):
                chosen_cont = outputs[2 * j].outputs[0].text.strip()
                rejected_cont = outputs[2 * j + 1].outputs[0].text.strip()
                rec = {
                    "image": str(p),
                    "label": label,
                    "prompt": PROMPT,
                    "chosen": f"{correct} {chosen_cont}",
                    "rejected": f"{wrong} {rejected_cont}",
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
    print(f"done -> {out_path} (root: {data_root_str})")


if __name__ == "__main__":
    main()
