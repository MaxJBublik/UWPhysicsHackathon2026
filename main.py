"""Batch runner for processed-circuit cross sections.

This script scans ``data/processed_circuits/`` for population sweep JSON files,
integrates the excitation probabilities in each file, and writes the resulting
merged cross-section tables to ``data/cross_sections/``.
"""

from __future__ import annotations

from pathlib import Path

from src.analysis.cross_sections import save_processed_circuit_cross_sections


def main() -> None:
    processed_dir = Path("data/processed_circuits")
    cross_section_dir = Path("data/cross_sections")
    cross_section_dir.mkdir(parents=True, exist_ok=True)

    processed_files = sorted(processed_dir.glob("populations_*.json"))
    if not processed_files:
        print(f"[!] No processed-circuit JSON files found in {processed_dir}")
        return

    for processed_file in processed_files:
        cross_section_name = processed_file.stem.replace("populations_", "cross_sections_") + ".json"
        cross_section_path = cross_section_dir / cross_section_name
        save_processed_circuit_cross_sections(processed_file, cross_section_path)
        print(f"[+] Saved {cross_section_path}")


if __name__ == "__main__":
    main()