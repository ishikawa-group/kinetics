#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from ase.db import connect
from kinetics.microkinetics.get_reaction_energy import get_reaction_energy
from kinetics.microkinetics.rate_nh3 import get_nh3_formation_rate


def get_row_by_unique_id(db, unique_id):
    """Find row by unique_id using multiple search methods."""
    for method in [
        lambda: list(db.select(f'unique_id="{unique_id}"')),
        lambda: list(db.select(unique_id=unique_id)),
        lambda: [r for r in db.select() if r.key_value_pairs.get('unique_id') == unique_id],
        lambda: [db.get(int(unique_id))]
    ]:
        try:
            rows = method()
            if rows:
                return rows[0]
        except (ValueError, KeyError):
            continue

    print(f"Error: unique_id '{unique_id}' not found in database")
    sys.exit(1)


def load_results(out_json):
    """Load existing results from JSON file."""
    if not (Path(out_json).exists() and Path(out_json).stat().st_size > 0):
        return [], []

    with open(out_json) as f:
        existing = json.load(f)
    results = existing if isinstance(existing, list) else [existing]
    ids = [entry.get('unique_id') for entry in results]
    return results, ids


def save_results(out_json, results):
    """Save results to JSON file."""
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--surf_db", required=True, help='表面構造を含むASE DB (JSON)')
    parser.add_argument("--out_json", required=True, help='結果出力用JSONファイル')
    parser.add_argument("--unique_id", required=True, help='処理する構造のユニークID')
    parser.add_argument("--base_dir", default="./result", help='計算用ベースディレクトリ')
    parser.add_argument("--overwrite", default=True, help='強制的に計算を実行')
    parser.add_argument("--log_level", default="INFO", help='ログレベル')
    parser.add_argument("--calculator", default="mace", help='計算機タイプ')
    parser.add_argument("--yaml_path", help='VASP設定YAMLファイルパス')
    args = parser.parse_args()

    surf_db = args.surf_db
    calculator = args.calculator
    unique_id = args.unique_id
    out_json = args.out_json

    # Setup
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    results, ids = load_results(out_json)

    # Initialize entry if not exists
    if unique_id not in ids:
        results.append({"unique_id": unique_id, "overpotential": None})
        save_results(out_json, results)
        print(f"Added initial entry for {unique_id}")

    # Get structure from database
    db = connect(surf_db)
    row = get_row_by_unique_id(db, unique_id)
    surf_atoms = row.toatoms(add_additional_information=True)
    d = surf_atoms.info.pop("data", {})
    surf_atoms.info["adsorbate_info"] = d["adsorbate_info"]

    reaction_file = "nh3decomposition.txt"
    energy_shift = [0] * 4

    # add vacuum region to surface
    vacuum = 7.0
    surf_atoms.center(vacuum=vacuum, axis=2)
    surf_atoms.translate([0, 0, -vacuum+0.1])

    deltaEs = get_reaction_energy(reaction_file=reaction_file, surface=surf_atoms,
                                  calculator=calculator, verbose=True, dirname="work")
    rate = get_nh3_formation_rate(deltaEs=deltaEs, reaction_file=reaction_file, rds=5)

    # Create result entry
    entry = {
        "unique_id": unique_id,
        "rate": rate,
        "bottom_deltaE": np.min(deltaEs),
        "chemical_formula": surf_atoms.get_chemical_formula()
    }

    # Update or append result
    for i, existing_entry in enumerate(results):
        if existing_entry.get('unique_id') == unique_id:
            results[i] = entry
            break
    else:
        results.append(entry)

    # Save and print results
    save_results(out_json, results)
    print(f"rate: {rate}")
