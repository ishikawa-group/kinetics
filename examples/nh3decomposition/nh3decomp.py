import argparse
import json
import logging
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from ase.db import connect
from ase import Atoms
from kinetics.microkinetics.get_reaction_energy import get_reaction_energy
from kinetics.microkinetics.rate_nh3 import get_nh3_formation_rate
from kinetics.utils import get_row_by_unique_id, plot_energy_diagram, fix_lower_surface


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--surf_db", default="structures.json", help='表面構造を含むASE DB (JSON)')
    parser.add_argument("--out_json", default="calc_result.json", help='結果出力用JSONファイル')
    parser.add_argument("--unique_id", help='処理する構造のユニークID')
    parser.add_argument("--base_dir", default="./result", help='計算用ベースディレクトリ')
    parser.add_argument("--overwrite", default=True, help='強制的に計算を実行')
    parser.add_argument("--log_level", default="INFO", help='ログレベル')
    parser.add_argument("--calculator", default="mace", help='計算機タイプ')
    parser.add_argument("--yaml_path", help='VASP設定YAMLファイルパス')
    args = parser.parse_args()

    surf_db = args.surf_db
    calculator = args.calculator
    out_json = args.out_json
    unique_id = args.unique_id
    yaml_path = args.yaml_path

    # If unique_id is not provided, use the first one from surf_db
    if not Path(surf_db).exists():
        raise ValueError(f"{surf_db} not exists.")

    if unique_id is None:
        db = connect(surf_db)
        first_row = next(db.select())
        tmp_id = first_row.key_value_pairs.get("unique_id")
        if tmp_id is None:
            unique_id = first_row.unique_id
        else:
            unique_id = tmp_id

    # Get structure from database
    db = connect(surf_db)
    row = get_row_by_unique_id(db, unique_id)
    surf = row.toatoms(add_additional_information=True)

    reaction_file = "nh3decomposition2.txt"
    energy_shift = [0] * 4

    # add vacuum region to surface
    vacuum = 7.0
    surf.center(vacuum=vacuum, axis=2)
    surf.translate([0, 0, -vacuum+0.1])
    surf = fix_lower_surface(atoms=surf)

    deltaEs = get_reaction_energy(reaction_file=reaction_file, surface=surf, calculator=calculator, 
                                  yaml_path=yaml_path, verbose=True, dirname="work", opt_steps=1)
                                  
    deltaEs = np.insert(deltaEs, 0, 0.0)   # add zero in the heading

    # rate = get_nh3_formation_rate(deltaEs=deltaEs, reaction_file=reaction_file, rds=5, debug=True)
    plot_energy_diagram(steps=range(len(deltaEs)), values=deltaEs, figname="deltaE.png",
                        labels=["NH$_3$", "NH$_3$*", "NH$_2$* + H*", "NH* + 2H*", "N* + 3H*",
                                "0.5N$_2$ + 3H*", "0.5N$_2$ + 1.5H$_2$"])

    print("Done")
