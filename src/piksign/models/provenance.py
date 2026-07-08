"""Provenance branch: C2PA / EXIF heuristics. Free, instant, rule-based.

What it can and cannot do:
  - A C2PA manifest naming OpenAI/Google => near-certain AI (score 1.0).
    Survives direct file sharing; stripped by most social platforms.
  - Full camera EXIF (make/model) is a WEAK real prior - it is trivially
    forgeable, so it only nudges the score, never overrides pixel evidence.
  - Absence of everything = unknown (score 0.5): the common laundered case.

This is a byte-level heuristic scan, not a cryptographic validation. If the
`c2patool` binary is on PATH it is used for real manifest parsing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

AI_GENERATOR_MARKERS = [
    b"openai", b"gpt-4o", b"gpt-image", b"dall-e", b"dalle",
    b"google ai", b"gemini", b"imagen", b"midjourney", b"stability",
    b"adobe firefly", b"bing image creator",
]
C2PA_MARKERS = [b"c2pa", b"jumb", b"jumd", b"contentauth", b"contentcredentials"]
SCAN_BYTES = 8 << 20  # metadata lives near the top of the file


def _c2patool(path: Path) -> dict | None:
    exe = shutil.which("c2patool")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, str(path), "--output", "json"],
            capture_output=True, timeout=30, text=True,
        )
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout)
    except Exception:  # noqa: BLE001
        pass
    return None


def check_file(path: str | Path) -> dict:
    path = Path(path)
    head = path.read_bytes()[:SCAN_BYTES]
    low = head.lower()

    has_c2pa = any(m in low for m in C2PA_MARKERS)
    generator = next((m.decode() for m in AI_GENERATOR_MARKERS if m in low), None)

    manifest = _c2patool(path) if has_c2pa else None
    if manifest is not None:
        txt = json.dumps(manifest).lower()
        gen2 = next((m.decode() for m in AI_GENERATOR_MARKERS if m.decode() in txt), None)
        generator = generator or gen2

    camera = None
    software = None
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            named = {TAGS.get(k, k): v for k, v in exif.items()}
            make, model = named.get("Make"), named.get("Model")
            software = named.get("Software")
            if make or model:
                camera = f"{make or ''} {model or ''}".strip()
    except Exception:  # noqa: BLE001
        pass

    if has_c2pa and generator:
        verdict, score = "ai", 1.0
    elif generator:  # generator string without C2PA structure - suspicious
        verdict, score = "lean_ai", 0.9
    elif camera:
        verdict, score = "lean_real", 0.35
    else:
        verdict, score = "unknown", 0.5

    return {
        "verdict": verdict,
        "score": score,  # calibrated like the other branches: P(fake)-ish
        "c2pa": has_c2pa,
        "generator_marker": generator,
        "camera_exif": camera,
        "software": str(software) if software else None,
    }
