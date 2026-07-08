"""piksign - check whether images are AI-generated.

    piksign check photo.jpg
    piksign check folder/ --vlm --json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .fusion import default_calibration, fuse, load_calibration
from .models.provenance import check_file
from .paths import IMG_EXTS, ckpt_root

Image.MAX_IMAGE_PIXELS = None


def collect_images(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXTS))
        elif p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def cmd_check(args: argparse.Namespace) -> None:
    import torch
    from .models.expert import PixelExpert, is_expert_checkpoint

    images = collect_images(args.paths)
    if not images:
        raise SystemExit("no images found")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = args.ckpt_root or ckpt_root()

    experts = {}
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if is_expert_checkpoint(d) and (not args.experts or d.name in args.experts):
                experts[d.name] = PixelExpert.load(d, device=device)
    vlm = None
    if args.vlm:
        from .models.vlm import SemanticVLM
        adapter = root / "vlm_dpo"
        vlm = SemanticVLM(adapter=adapter if adapter.exists() else None, device=device)

    calib = load_calibration(root / "calibration.json")
    if calib is None:
        calib = default_calibration(list(experts) + (["vlm"] if vlm else []))
        if not args.json:
            print("NOTE: no calibration.json found - using naive 0.5 thresholds\n")

    results = []
    for path in images:
        scores: dict[str, float] = {}
        detail: dict = {}
        prov = check_file(path)
        scores["provenance"] = prov["score"]
        detail["provenance"] = prov
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")
        except Exception as e:  # noqa: BLE001
            results.append({"path": str(path), "error": str(e)})
            continue
        for name, expert in experts.items():
            scores[name] = expert.score_pil(img, n_crops=args.crops, agg=args.agg)
        if vlm:
            v = vlm.score(img, with_rationale=args.rationale)
            scores["vlm"] = v["prob_fake"]
            detail["vlm"] = v
        fused = fuse(scores, calib)
        results.append({"path": str(path), **fused, "detail": detail})

    if args.json:
        print(json.dumps(results, indent=2))
        return
    for r in results:
        if "error" in r:
            print(f"{r['path']}: ERROR {r['error']}")
            continue
        verdict = "AI-GENERATED" if r["is_ai"] else "no AI evidence"
        print(f"\n{r['path']}")
        print(f"  verdict : {verdict}  (strength {r['strength']:+.2f})")
        if r["fired_branches"]:
            print(f"  fired   : {', '.join(r['fired_branches'])}")
        for b, s in r["scores"].items():
            thr = r["thresholds"][b]
            mark = " <-- fired" if s >= thr else ""
            print(f"    {b:<16} {s:.3f} (thr {thr:.3f}){mark}")
        rat = r["detail"].get("vlm", {}).get("rationale")
        if rat:
            print(f"  vlm     : {rat}")
        prov = r["detail"]["provenance"]
        if prov["generator_marker"] or prov["camera_exif"]:
            print(f"  metadata: generator={prov['generator_marker']} camera={prov['camera_exif']}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="piksign", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="check image(s) for AI generation")
    c.add_argument("paths", nargs="+", type=Path)
    c.add_argument("--experts", nargs="*", default=None, help="subset of expert checkpoint names")
    c.add_argument("--ckpt-root", type=Path, default=None)
    c.add_argument("--vlm", action="store_true", help="also run the semantic VLM branch")
    c.add_argument("--rationale", action="store_true", help="generate VLM explanation text")
    c.add_argument("--crops", type=int, default=10)
    c.add_argument("--agg", choices=["max", "mean", "top3"], default="top3")
    c.add_argument("--json", action="store_true")
    c.set_defaults(fn=cmd_check)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
