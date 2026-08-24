r"""
One command to answer "am I running the current code?" -- no grepping for
specific strings, no guessing which folder is stale.

    python scripts\check_version.py

Prints the resolved __file__ path and version stamp for every module that
matters, plus a hash of each file's contents so you can compare two folders
byte-for-byte without a diff tool.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)


def file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def main() -> int:
    print(f"Repo root: {REPO_ROOT}\n")

    ok = True

    try:
        import src.quantum.time_evolution as te
        stamp = getattr(te, "_MODULE_VERSION", "<MISSING - STALE FILE>")
        has_backend = "backend" in te.CollisionDynamicsSimulator.__init__.__code__.co_varnames
        print(f"time_evolution.py")
        print(f"  path    : {te.__file__}")
        print(f"  version : {stamp}")
        print(f"  hash    : {file_hash(te.__file__)}")
        print(f"  has 'backend' kwarg on CollisionDynamicsSimulator: {has_backend}")
        if stamp == "<MISSING - STALE FILE>" or not has_backend:
            ok = False
    except Exception as exc:
        print(f"[!] Failed to import src.quantum.time_evolution: {exc}")
        ok = False

    print()
    try:
        import src.quantum.time_evolution_gpu as teg
        has_arch = hasattr(teg, "cuda_arch_status")
        print(f"time_evolution_gpu.py")
        print(f"  path    : {teg.__file__}")
        print(f"  hash    : {file_hash(teg.__file__)}")
        print(f"  has cuda_arch_status: {has_arch}")
        if not has_arch:
            ok = False
    except Exception as exc:
        print(f"[!] Failed to import src.quantum.time_evolution_gpu: {exc}")
        ok = False

    print()
    try:
        import src.quantum.mapper as mp
        m_src = Path(mp.__file__).read_text()
        has_ev_attr = "self.unphysical_penalty_ev = unphysical_penalty_ev" in m_src
        print(f"mapper.py")
        print(f"  path    : {mp.__file__}")
        print(f"  hash    : {file_hash(mp.__file__)}")
        print(f"  summary() attribute bug fixed: {has_ev_attr}")
        if not has_ev_attr:
            ok = False
    except Exception as exc:
        print(f"[!] Failed to import src.quantum.mapper: {exc}")
        ok = False

    print("\n" + "=" * 60)
    print("ALL CURRENT" if ok else "STALE FILE(S) DETECTED -- see [!]/False lines above")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
