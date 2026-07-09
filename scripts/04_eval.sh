#!/usr/bin/env bash
# Evaluation: assemble reals, build degraded twins, score, calibrate, report.
# Fresh fakes must exist already (gen_eval_api needs your API keys):
#   python -m piksign.download.gen_eval_api --provider gemini --n 400 --yes
#   python -m piksign.download.gen_eval_api --provider openai --n 400 --yes
set -euo pipefail
cd "$(dirname "$0")/.."
E="${PIKSIGN_DATA:-data}/eval"

# eval reals: COCO val (+ add --phone /path/to/your/phone/photos when you have them)
python -m piksign.eval.harness assemble \
    --coco-val "${PIKSIGN_DATA:-data}/raw/coco/val2017_all" --n-coco 1500

# laundered twins of the entire clean tree
for preset in whatsapp double_jpeg screenshot double_resize; do
    python -m piksign.launder --input "$E/clean" --output "$E/$preset" --preset "$preset"
done

# score everything with all experts + provenance + VLM
python -m piksign.eval.harness run --vlm

# calibrate thresholds to a 5% global FPR budget, then report
python -m piksign.fusion --scores "$E/scores.jsonl" --global-fpr 0.05
python -m piksign.eval.harness report

echo "evaluation complete. see $E/report.json"
