"""Fetch the user's piksign_data folder from Google Drive (manifests, sft.jsonl, zips).

    python -m piksign.download.drive [--include-zips]
"""
from __future__ import annotations

import argparse

import gdown

from ..paths import ensure, raw_dir

FOLDER_URL = "https://drive.google.com/drive/folders/1lRuvdKigo_C3dZsEMMJ5f4ykbjq8EZWN"
FILE_IDS = {
    "manifest_pico_train.csv": "1z4QqDE8JoU4-_x8Cd86J2v9eQPwDZ8Dx",
    "manifest_pico_val.csv": "1ZMf5I9lsJss_AGeTcDCmUrxraGXx8eY4",
    "sft.jsonl": "12AV1JsO4ERwfDC2XLp6V138DzGv8T3XN",
    "neg.zip": "1o26T5Vi5DoRw4reLBF0dDnyTPzdcNGWT",
    "pos.zip": "1Z5yd3V2Dggg__Gmx6PofkY4DIc1oNj_E",
}
SMALL = ["manifest_pico_train.csv", "manifest_pico_val.csv", "sft.jsonl"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--include-zips", action="store_true",
        help="also fetch pos.zip/neg.zip (not needed: we re-download clean copies)",
    )
    args = ap.parse_args()

    out = ensure(raw_dir("piksign_drive"))
    names = list(FILE_IDS) if args.include_zips else SMALL
    for name in names:
        dest = out / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {dest} exists")
            continue
        gdown.download(id=FILE_IDS[name], output=str(dest), quiet=False)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
