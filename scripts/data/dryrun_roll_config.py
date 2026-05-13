"""Dry-run the ROLL RLVR config: YAML parse + field sanity check.

DEV image does not have hydra/dacite/torch installed, so we do the
pragmatic subset:
- Parse YAML
- Resolve ${...} interpolations with OmegaConf (lightweight)
- Diff the top-level keys against the RLVRConfig dataclass source to
  catch typos / removed fields before the TRAIN round-trip.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-path", default="configs/tombench-rlvr")
    p.add_argument("--config-name", default="rlvr_config_stage1")
    p.add_argument("--roll-config-source",
                   default="framework/ROLL/roll/pipeline/rlvr/rlvr_config.py")
    args = p.parse_args()

    # 1. YAML parse + resolve
    try:
        from omegaconf import OmegaConf
    except ImportError:
        print("omegaconf not installed in DEV image; adding to requirements.txt would enable this.")
        return 0

    cfg_path = Path(args.config_path) / f"{args.config_name}.yaml"
    cfg = OmegaConf.load(cfg_path)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    print(f"Loaded and resolved {cfg_path}")

    # 2. Extract fields declared in RLVRConfig source (best-effort regex).
    src = Path(args.roll_config_source).read_text()
    # Match lines like "field_name: type = ..." or "field_name: type"
    decl_pattern = re.compile(r"^\s{4}(\w+)\s*:\s*[A-Za-z0-9_\[\], .'\"|]+(?:=|$)", re.MULTILINE)
    declared = set(decl_pattern.findall(src))
    if not declared:
        print("(could not extract RLVRConfig fields from source; skipping diff)")
    else:
        print(f"RLVRConfig declares {len(declared)} top-level fields.")

    # 3. Compare top-level cfg keys to declared fields
    cfg_keys = set(resolved.keys())
    # Some keys are expected (hydra, system_envs etc) — ignore those.
    expected_non_fields = {"hydra", "system_envs", "tracker_kwargs", "validation",
                           "checkpoint_config"}
    unknown = cfg_keys - declared - expected_non_fields
    if declared and unknown:
        print(f"Config has {len(unknown)} keys that don't match any RLVRConfig field:")
        for k in sorted(unknown):
            print(f"  - {k}")
        print("(this may be OK if ROLL reads them via dynamic dict access or if our"
              " regex missed them — verify on TRAIN).")
    else:
        print("All top-level cfg keys match RLVRConfig dataclass fields or expected extras.")

    # 4. Spot-check reward worker class path exists on disk (ROLL mounted)
    rewards = resolved.get("rewards", {})
    for name, rc in rewards.items():
        worker_cls = rc.get("worker_cls", "")
        if not worker_cls:
            continue
        # Convert "roll.pipeline.rlvr.rewards.tom_mcq_reward_worker.TomMcqRewardWorker"
        # to "framework/ROLL/roll/pipeline/rlvr/rewards/tom_mcq_reward_worker.py"
        parts = worker_cls.split(".")
        module_path = Path("framework/ROLL") / "/".join(parts[:-1])
        py_path = module_path.with_suffix(".py")
        cls_name = parts[-1]
        if py_path.exists():
            body = py_path.read_text()
            if f"class {cls_name}" in body:
                print(f"  reward[{name}]: {worker_cls} -> OK ({py_path})")
            else:
                print(f"  reward[{name}]: {worker_cls} -> module exists but class {cls_name!r} not found")
        else:
            print(f"  reward[{name}]: {worker_cls} -> FILE NOT FOUND: {py_path}")

    # 5. Spot-check data file paths that would be bind-mounted to /mnt/data
    file_names = resolved.get("actor_train", {}).get("data_args", {}).get("file_name", [])
    print(f"\nactor_train data files (inside container):")
    for f in file_names:
        # /mnt/data/tom_train_4k.jsonl -> data/tom/tom_train_4k.jsonl (on DEV host)
        if f.startswith("/mnt/data/"):
            host = Path("data/tom") / f[len("/mnt/data/"):]
            exists = host.exists()
            size = host.stat().st_size if exists else 0
            print(f"  {f} -> DEV has {host} ({'OK' if exists else 'MISSING'}, {size} bytes)")

    val_files = resolved.get("validation", {}).get("data_args", {}).get("file_name", [])
    print(f"\nvalidation data files:")
    for f in val_files:
        if f.startswith("/mnt/data/"):
            host = Path("data/tom") / f[len("/mnt/data/"):]
            exists = host.exists()
            size = host.stat().st_size if exists else 0
            print(f"  {f} -> DEV has {host} ({'OK' if exists else 'MISSING'}, {size} bytes)")

    print("\nDone (lightweight DEV check). Full dacite/hydra validation happens on TRAIN.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

