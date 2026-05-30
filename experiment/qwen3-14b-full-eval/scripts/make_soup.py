"""Weight-space model soup (WiSE-FT): out = (1-alpha)*base + alpha*ft.

Memory-bounded: processes one output shard at a time, pulling each tensor from
base/ft via mmap (safetensors safe_open). base and ft may have different shard
counts; only the tensor NAMES must match (verified: 443/443 for Qwen3-14B base
vs v3.1).

Usage (inside the serve image, which ships torch+safetensors):
  python make_soup.py --base /path/Qwen3-14B --ft /path/v31 --alpha 0.5 --out /path/soup50
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--alpha", type=float, required=True, help="weight on ft (0=base, 1=ft)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    a = args.alpha
    base_dir, ft_dir, out_dir = Path(args.base), Path(args.ft), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ft_index = json.loads((ft_dir / "model.safetensors.index.json").read_text())
    ft_map = ft_index["weight_map"]                      # name -> ft shard file
    base_index = json.loads((base_dir / "model.safetensors.index.json").read_text())
    base_map = base_index["weight_map"]                  # name -> base shard file

    # group output tensors by their ft shard (mirror ft sharding)
    out_shards = defaultdict(list)
    for name, shard in ft_map.items():
        out_shards[shard].append(name)

    # lazy file handles
    _open = {}
    def handle(d: Path, fname: str):
        key = str(d / fname)
        if key not in _open:
            _open[key] = safe_open(str(d / fname), framework="pt", device="cpu")
        return _open[key]

    miss = [n for n in ft_map if n not in base_map]
    if miss:
        raise SystemExit(f"{len(miss)} tensors missing from base, e.g. {miss[:3]}")

    n_done = 0
    for shard_file, names in out_shards.items():
        tensors = {}
        for name in names:
            bt = handle(base_dir, base_map[name]).get_tensor(name)
            ft = handle(ft_dir, ft_map[name]).get_tensor(name)
            dt = bt.dtype
            out = (1.0 - a) * bt.to(torch.float32) + a * ft.to(torch.float32)
            tensors[name] = out.to(dt).contiguous()
            n_done += 1
        save_file(tensors, str(out_dir / shard_file), metadata={"format": "pt"})
        print(f"  wrote {shard_file}  ({len(names)} tensors, {n_done} total)", flush=True)
        del tensors

    # copy index + all aux files (config, tokenizer, chat_template, generation_config)
    # from ft so chat template / special tokens match the trained model.
    shutil.copy(ft_dir / "model.safetensors.index.json", out_dir / "model.safetensors.index.json")
    for f in ft_dir.iterdir():
        if f.suffix == ".safetensors" or f.name == "model.safetensors.index.json":
            continue
        if f.is_file():
            shutil.copy(f, out_dir / f.name)
    print(f"DONE soup alpha={a} -> {out_dir}  ({n_done} tensors)", flush=True)


if __name__ == "__main__":
    main()
