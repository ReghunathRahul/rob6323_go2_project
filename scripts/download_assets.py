#!/usr/bin/env python3
import os
from pathlib import Path
import urllib.request

USDC_URL = "https://drive.google.com/uc?export=download&id=1yKxWCZjPWknPWzr62dmj92uIKQVpm8y-"
FILENAME = "uneven_terrain.usdc"

def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    assets_dir = repo_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    target_path = assets_dir / FILENAME

    if target_path.exists():
        print(f"File already exists at {target_path}, skipping download.")
        return

    print(f"Downloading {USDC_URL} -> {target_path}")
    urllib.request.urlretrieve(USDC_URL, target_path.as_posix())
    print("Download complete.")

if __name__ == "__main__":
    main()
