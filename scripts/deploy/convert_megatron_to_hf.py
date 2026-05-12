"""Convert ROLL/Megatron checkpoint → HuggingFace format for vLLM serving.

ROLL ships `mcore_adapter` to handle this. This script is a thin CLI wrapper.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Megatron checkpoint dir")
    p.add_argument("--dst", required=True, help="HF-format output dir")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B",
                   help="reference HF model for tokenizer/config")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit(f"src {src} missing")
    dst.mkdir(parents=True, exist_ok=True)

    # Use ROLL's mcore_adapter conversion script
    converter = Path("framework/ROLL/mcore_adapter/scripts/convert_to_hf.py")
    if converter.exists():
        print(f"[convert] using {converter}")
        subprocess.run([
            "python", str(converter),
            "--load", str(src),
            "--save", str(dst),
            "--base-model", args.base_model,
        ], check=True)
    else:
        # Fallback: if final ckpt is already HF (some ROLL strategies), just copy
        config = src / "config.json"
        if config.exists():
            print(f"[convert] {src} appears to be HF format already; copying")
            for item in src.iterdir():
                d = dst / item.name
                if item.is_dir():
                    shutil.copytree(item, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, d)
        else:
            raise SystemExit(
                f"converter {converter} not found and {src} is not HF format. "
                "Install ROLL mcore_adapter or manually convert."
            )
    print(f"[convert] done → {dst}")


if __name__ == "__main__":
    main()
