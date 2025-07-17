from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
from ase import Atoms
from ase.build import fcc111, add_adsorbate, molecule
from ase.io import read, write
from kinetics.utils import (setup_logging, set_initial_magmoms, set_calculator, convert_numpy_types,
                            parallel_displacement, fix_lower_surface, plot_free_energy_diagram)

YAML_PATH = os.path.join(Path(__file__).parent, "data", "vasp.yaml")

ADSORPTION_SITES = {
    "fcc": [2/3, 1/3, 0],
    "hcp": [1/3, 2/3, 0],
    "top": [0, 0, 0],
    "bridge": [0.5, 0, 0],
    "hollow": [0.5, 0.5, 0],
}

MOLECULES = {
    "OH":  Atoms("OH",  positions=[(0, 0, 0), (0, 0, 0.97)]),
    "H2O": Atoms("OHH", positions=[(0, 0, 0), (0.759, 0, 0.588), (-0.759, 0, 0.588)]),
    "HO2": Atoms("OOH", positions=[(0, 0, 0), (0, -0.73, 1.264), (0.939, -0.8525, 1.4766)]),
    "OOH": Atoms("OOH", positions=[(0, 0, 0), (0, -0.73, 1.264), (0.939, -0.8525, 1.4766)]),  # alias
    "H2":  Atoms("HH",  positions=[(0, 0, 0), (0, 0, 0.74)]),
    "O2":  Atoms("OO",  positions=[(0, 0, 0), (0, 0, 1.21)]),
    "H":   Atoms("H",   positions=[(0, 0, 0)]),
    "O":   Atoms("O",   positions=[(0, 0, 0)])
}

CLOSED_SHELL_MOLECULES = ["H2", "H2O"]

SLAB_VACUUM = 8.0
GAS_BOX = 15.0
ADSORBATE_HEIGHT = 2.0


def save_structure_and_energy(atoms: Atoms, energy: float, filepath: str):
    """Save structure and energy to file."""
    write(filepath, atoms)
    with open(filepath.replace('.xyz', '_energy.txt'), 'w') as f:
        f.write(f"{energy:.6f}\n")


def load_structure_and_energy(filepath: str) -> Tuple[Atoms, float]:
    """Load structure and energy from file."""
    atoms = read(filepath)
    energy_file = filepath.replace('.xyz', '_energy.txt')
    if os.path.exists(energy_file):
        with open(energy_file, 'r') as f:
            energy = float(f.read().strip())
    else:
        energy = 0.0
    return atoms, energy


def optimize_gas_molecule(molecule_name: str, gas_box_size: float, work_directory: str,
                          calculator: str = "mace", yaml_path: str = YAML_PATH) -> Tuple[Atoms, float]:
    """Optimize gas phase molecule."""
    molecule = MOLECULES[molecule_name].copy()
    molecule.set_cell([gas_box_size, gas_box_size, gas_box_size])
    molecule.set_pbc(True)
    molecule.center()

    molecule = set_initial_magmoms(molecule)
    optimized_molecule = set_calculator(atoms=molecule, kind="gas", calculator=calculator,
                                        yaml_path=yaml_path, calc_directory=work_directory)
    
    if molecule_name in CLOSED_SHELL_MOLECULES:
        calc = optimized_molecule.calc
        calc.set(ispin=1)
        optimized_molecule.set_calculator(calc)
    
    return optimized_molecule, optimized_molecule.get_potential_energy()


def optimize_bulk(bulk: Atoms, work_directory: str, calculator_type: str = "mace",
                  yaml_path: str = None) -> Tuple[Atoms, float]:
    """Optimize bulk structure."""
    work_dir = Path(work_directory)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = work_dir / "optimized_bulk.xyz"
    if result_file.exists():
        return load_structure_and_energy(str(result_file))
    
    bulk_structure = bulk.copy()
    bulk_structure.set_pbc(True)
    bulk_structure = set_initial_magmoms(bulk_structure)
    
    optimized_bulk = set_calculator(
        bulk_structure, "bulk",
        calculator=calculator_type,
        yaml_path=yaml_path,
        calc_directory=work_directory
    )
    
    energy = optimized_bulk.get_potential_energy()
    
    save_structure_and_energy(optimized_bulk, energy, str(result_file))
    return optimized_bulk, energy


def optimize_slab(bulk: Atoms, work_directory: str, calculator_type: str = "mace",
                  yaml_path: str = None) -> Tuple[Atoms, float]:
    """Optimize slab structure from bulk."""
    work_dir = Path(work_directory)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = work_dir / "optimized_slab.xyz"
    if result_file.exists():
        return load_structure_and_energy(str(result_file))
    
    # Create slab from bulk
    slab = fcc111(bulk.get_chemical_symbols()[0], size=(3, 3, 4), a=bulk.cell.lengths()[0])
    slab.set_pbc(True)
    slab = parallel_displacement(slab, vacuum=SLAB_VACUUM)
    slab = fix_lower_surface(slab)
    slab = set_initial_magmoms(slab)
    
    optimized_slab = set_calculator(
        slab, "surf",
        calculator=calculator_type,
        yaml_path=yaml_path,
        calc_directory=work_directory
    )
    
    energy = optimized_slab.get_potential_energy()
    
    save_structure_and_energy(optimized_slab, energy, str(result_file))
    return optimized_slab, energy


def get_adsorption_energy(slab: Atoms, adsorbate: Atoms,
                          position_spec: Union[str, Tuple[float, float], List[int]],
                          work_directory: str, calculator_type: str = "mace", yaml_path: str = None,
                          height: float = None, orientation: str = None) -> Tuple[float, float]:
    """Unified adsorption calculation for different position specifications."""
    work_dir = Path(work_directory)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = work_dir / "result.json"
    if result_file.exists():
        with open(result_file, "r") as f:
            data = json.load(f)
        return data["E_total"]
    
    # Determine adsorption position
    if isinstance(position_spec, str):
        # Site-based adsorption
        if position_spec not in ADSORPTION_SITES:
            raise ValueError(f"Unknown adsorption site: {position_spec}")
        site_coord = ADSORPTION_SITES[position_spec]
        cell = slab.get_cell()
        position = site_coord @ cell[:2, :2]
    elif isinstance(position_spec, (tuple, list)) and len(position_spec) == 2:
        # Offset-based adsorption
        position = np.array(position_spec)
    elif isinstance(position_spec, list) and all(isinstance(x, int) for x in position_spec):
        # Index-based adsorption
        positions = slab.get_positions()
        position = np.mean(positions[position_spec], axis=0)[:2]
    else:
        raise ValueError(f"Invalid position specification: {position_spec}, {type(position_spec)}")
    
    # Add adsorbate
    ads_and_slab = slab.copy()
    if height is None:
        height = 2.0
    
    add_adsorbate(ads_and_slab, adsorbate, height=height, position=position)
    set_initial_magmoms(ads_and_slab)
    
    ads_and_slab = set_calculator(ads_and_slab, "surf", calculator=calculator_type,
                                  yaml_path=yaml_path, calc_directory=work_directory)

    energy = ads_and_slab.get_potential_energy()
    
    # Save results
    save_structure_and_energy(ads_and_slab, energy, str(work_dir / "optimized.xyz"))
    with open(result_file, 'w') as f:
        json.dump({"E_total": energy}, f)
    
    return energy


def get_deltaEs(slab: Atoms, E_slab: float, base_dir: Path, calculator: str = "mace",
                yaml_path: str = None, adsorbates: Dict[str, List] = None,
                overwrite: bool = False) -> Tuple[List[float], Dict[str, float]]:
    """Calculate energies for all adsorbates and compute ORR reaction energies."""
    logger = setup_logging()
    E = {}
    
    # Gas-phase molecules
    for molecule in ["O2", "H2", "H2O", "OH", "HO2", "O"]:
        mol_dir = base_dir / f"gas_{molecule}"
        if not overwrite and (mol_dir / "result.json").exists():
            with open(mol_dir / "result.json", 'r') as f:
                E[f"gas_{molecule}"] = json.load(f)["E_total"]
        else:
            _, energy = optimize_gas_molecule(molecule, GAS_BOX, str(mol_dir), calculator, yaml_path)
            E[f"gas_{molecule}"] = energy
    
    # Adsorbed molecules
    default_adsorbates = {
        "OOH": [ADSORPTION_SITES["fcc"][:2]],
        "O": [ADSORPTION_SITES["fcc"][:2]],
        "OH": [ADSORPTION_SITES["fcc"][:2]]
    }
    adsorbates = adsorbates or default_adsorbates

    # find adsorption positions with lowest adsorption energy
    for adsorbate in adsorbates:
        adsorption_energies = []
        for i, position in enumerate(adsorbates[adsorbate]):
            ads_dir = str(base_dir / f"{adsorbate}surf_{i:02d}")
            past_json = Path(ads_dir) / "result.json"

            if past_json.exists() and not overwrite:
                with open(past_json, "r") as f:
                    energy = json.load(f)["E_total"]
            else:
                energy = get_adsorption_energy(slab, MOLECULES[adsorbate], position, ads_dir,
                                               calculator, yaml_path)

            adsorption_energies.append(energy)

        E[f"{adsorbate}surf"] = min(adsorption_energies)
    
    # Add slab energy
    E["surf"] = E_slab
    
    # Compute ORR reaction energies
    Es = {
        "O2": E["gas_O2"],
        "H2": E["gas_H2"],
        "H2O": E["gas_H2O"],
        "Osurf": E["Osurf"],
        "OHsurf": E["OHsurf"],
        "OOHsurf": E["OOHsurf"],
        "surf": E_slab
    }
    
    # --- ORR reaction steps ---
    # 1) O2 + * + H+ + e- → OOH*
    # 2) OOH* + H+ + e- → O* + H2O
    # 3) O* + H+ + e- → OH*
    # 4) OH* + H+ + e- → * + H2O
    deltaEs = np.zeros(4)
    deltaEs[0] = Es["OOHsurf"] - Es["surf"] - Es["O2"] - 0.5 * Es["H2"]
    deltaEs[1] = Es["Osurf"] + Es["H2O"] - Es["OOHsurf"] - 0.5 * Es["H2"]
    deltaEs[2] = Es["OHsurf"] - Es["Osurf"] - 0.5 * Es["H2"]
    deltaEs[3] = Es["surf"] + Es["H2O"] - Es["OHsurf"] - 0.5 * Es["H2"]
    
    return deltaEs, Es


def get_orr_overpotential(bulk: Atoms = None, deltaEs: List[float] = None, T: float = 298.15,
                          work_dir: Union[str, Path] = "result/matter_sim", 
                          overwrite: bool = False, log_level: str = "INFO", 
                          calculator: str = "mace", verbose: bool = True, 
                          save_plot: bool = False, adsorbates: Dict[str, List[Tuple[float, float]]] = None,
                          yaml_path: str = None) -> Dict[str, Any]:
    """Calculate ORR overpotential from bulk structure or reaction energies.
    
    Args:
        bulk: Bulk structure (required for full calculation)
        deltaEs: Pre-calculated reaction energies (alternative to bulk)
        T: Temperature [K]
        work_dir: Output directory
        overwrite: Overwrite existing results
        log_level: Logging level
        calculator: Calculator type
        verbose: Verbose output
        save_plot: Save free energy diagram
        adsorbates: Custom adsorbate positions
        yaml_path: VASP parameters file
    
    Returns:
        Dictionary with overpotential results
    """
    logger = setup_logging(log_level)
    base_path = Path(work_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Full calculation from bulk structure
    if bulk is None:
        raise ValueError("Either bulk structure or reaction_energies must be provided")
    
    logger.info("Starting ORR overpotential calculation from bulk structure...")
    
    # 1. Optimize bulk structure
    logger.info("Optimizing bulk structure...")
    bulk_dir = str(base_path / "bulk")
    optimized_bulk, bulk_energy = optimize_bulk(bulk, bulk_dir, calculator, yaml_path)
    
    # 2. Optimize slab structure
    logger.info("Optimizing surface...")
    slab_dir = str(base_path / "surf")
    optimized_slab, E_slab = optimize_slab(optimized_bulk, slab_dir, calculator, yaml_path)
    
    # 3. Calculate all required molecules and reaction energies
    logger.info("Calculating required molecules and reaction energies...")
    deltaEs, Es = get_deltaEs(optimized_slab, E_slab, base_path, calculator, yaml_path, adsorbates, overwrite)

    # 4. Convert to Gibbs energies and calculate overpotential
    num_rxn = 4  # 4-electron pathway
    assert len(deltaEs) == num_rxn, "reaction_energies must contain 4 elements"

    # Zero-point energy corrections (eV)-- Reference: https://doi.org/10.1021/ja405997s
    ZPE = {
        "H2": 0.35, "H2O": 0.57,
        "OOH": 0.44,  "OH": 0.37,  # temporary value
        "Osurf": 0.06, "OHsurf": 0.37, "OOHsurf": 0.44,
    }

    # Entropy terms T*S (eV) -- Reference: https://doi.org/10.1021/ja405997s
    S = {
        "H2": 0.403 / T, "H2O": 0.67 / T,
        "OOH": 0.20 / T, "OH": 0.20 / T,  # temporary value
        "Osurf": 0.0, "OHsurf": 0.0, "OOHsurf": 0.0,
    }

    # Calculate O2 corrections
    ZPE["O2"] = 2 * (ZPE["H2O"] - ZPE["H2"])
    S["O2"] = 2 * (S["H2O"] - S["H2"])

    # Calculate binding energies
    E_bind = {}
    E_bind["OOH"] = Es["OOHsurf"] - (Es["surf"] + (2 * Es["H2O"] - 1.5 * Es["H2"]))
    E_bind["OH"]  = Es["OHsurf"] - (Es["surf"] + (Es["H2O"] - 0.5 * Es["H2"]))
    E_bind["O"]   = Es["Osurf"] - (Es["surf"] + (Es["H2O"] - Es["H2"]))
    print("binding energies: ", E_bind)

    # Calculate binding Gibbs energies
    G_bind = {}
    G_bind["OOH"] = E_bind["OOH"] + (ZPE["OOHsurf"] - ZPE["OOH"]) - T * (S["OOHsurf"] - S["OOH"])
    G_bind["OH"]  = E_bind["OH"]  + (ZPE["OHsurf"] - ZPE["OH"]) - T * (S["OHsurf"] - S["OH"])
    G_bind["O"]   = E_bind["O"]   + (ZPE["Osurf"] - 0.5 * ZPE["O2"]) - T * (S["Osurf"] - 0.5 * S["O2"])
    print("binding Gibbs energies: ", G_bind)

    # Calculate ZPE and entropy corrections for each reaction step
    deltaZPEs = np.zeros(num_rxn)
    deltaTSs = np.zeros(num_rxn)

    deltaZPEs[0] = ZPE["OOHsurf"] + (-0.5 * ZPE["H2"] + - ZPE["O2"])
    deltaZPEs[1] = ZPE["Osurf"] + ZPE["H2O"] - ZPE["OOHsurf"] - 0.5 * ZPE["H2"]
    deltaZPEs[2] = ZPE["OHsurf"] - ZPE["Osurf"] - 0.5 * ZPE["H2"]
    deltaZPEs[3] = ZPE["H2O"] - ZPE["OHsurf"] - 0.5 * ZPE["H2"]

    deltaTSs[0] = T * S["OOHsurf"] + (-0.5 * T * S["H2"] - T * S["O2"])
    deltaTSs[1] = T * S["Osurf"] + T * S["H2O"] - T * S["OOHsurf"] - 0.5 * T * S["H2"]
    deltaTSs[2] = T * S["OHsurf"] - T * S["Osurf"] - 0.5 * T * S["H2"]
    deltaTSs[3] = T * S["H2O"] - T * S["OHsurf"] - 0.5 * T * S["H2"]

    # Calculate Gibbs energies
    deltaEs = np.array(deltaEs)
    deltaGs_u0 = deltaEs + deltaZPEs - deltaTSs  # ΔG at U=0 V

    # Free energy profiles
    G_profile_u0 = np.concatenate(([0.0], np.cumsum(deltaGs_u0)))
    U_eq = 1.23  # equilibrium potential [V]

    # Calculate step-wise Gibbs energy changes
    diff_G_u0 = np.diff(G_profile_u0)

    # find limiting potential (U_L) and overpotential (eta)
    dG_orr_max = np.max(diff_G_u0)
    U_L = (-1) * dG_orr_max
    eta = U_eq - U_L

    # Calculate profiles for U = 1.23 V and U = UL
    G_profile_ueq = G_profile_u0 - np.arange(num_rxn + 1) * (-1) * U_eq
    G_profile_ul = G_profile_u0 - np.arange(num_rxn + 1) * (-1) * U_L
    diff_G_eq = np.diff(G_profile_ueq)
    diff_G_ul = np.diff(G_profile_ul)

    if verbose:
        logger.info(f"Reaction energies: {deltaEs}")
        logger.info(f"Gibbs energies (at U = 0): {deltaGs_u0}")
        logger.info(f"Overpotential: {eta:.3f} V")
    
    if save_plot:
        plot_free_energy_diagram(deltaGs=deltaGs_u0, base_path=base_path,
                                 steps=["O2 + * + H+ + e-", "OOH*", "O* + H2O", "OH*", "* + H2O"])
    
    # 5. Save summary
    summary_file = base_path / "ORR_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("--- ORR Summary ---\n\n")
        f.write("total energies (eV)\n")
        f.write(json.dumps(convert_numpy_types(Es), indent=2))
        f.write(f"\n\ndeltaEs (eV): {', '.join(f'{e:+.3f}' for e in deltaEs)}\n")
        f.write(f"\n\ndeltaGs (eV): {', '.join(f'{e:+.3f}' for e in deltaGs_u0)}\n")
        f.write(f"limiting potential: {U_L:.3f} V\n")
        f.write(f"overpotential: {eta:.3f} V\n")
    
    logger.info(f"ORR calculation completed. Overpotential: {eta:.3f} V")

    return {
        "eta": eta,
        "reaction_energies": deltaEs,
        "Gibbs_energies": deltaGs_u0,
        "diffG_U0": diff_G_u0.tolist(),
        "diffG_eq": diff_G_eq.tolist(),
        "U_L": U_L,
        "G_profile_U_0": G_profile_u0.tolist(),
        "G_profile_U_eq": G_profile_ueq.tolist(),
        "G_profile_U_L": G_profile_ul.tolist()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORR Overpotential Calculator")
    parser.add_argument("--bulk", type=str, required=True, help="Bulk structure file")
    parser.add_argument("--outdir", type=str, default="result", help="Output directory")
    parser.add_argument("--calculator", type=str, default="mace", choices=["mace", "vasp"], help="calculator")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()
    
    # Load bulk structure
    bulk = read(args.bulk)
    
    # Run calculation
    results = get_orr_overpotential(bulk=bulk, work_dir=args.outdir, overwrite=args.overwrite,
                                    log_level=args.log_level, calculator=args.calculator)

    print(f"ORR Overpotential: {results['eta']:.3f} V")
