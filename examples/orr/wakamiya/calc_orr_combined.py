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
from kinetics.utils import setup_logging, set_initial_magmoms, my_calculator, convert_numpy_types, parallel_displacement, fix_lower_surface

# Constants
YAML_PATH = os.path.join(Path(__file__).parent, "data", "vasp.yaml")
ADSORPTION_SITES = {
    "fcc": np.array([2/3, 1/3, 0]),
    "hcp": np.array([1/3, 2/3, 0]),
    "top": np.array([0, 0, 0]),
    "bridge": np.array([0.5, 0, 0]),
    "hollow": np.array([0.5, 0.5, 0]),
}

MOLECULES = {
    "OH":  Atoms("OH",  positions=[(0, 0, 0), (0, 0, 0.97)]),
    "H2O": Atoms("OHH", positions=[(0, 0, 0), (0.759, 0, 0.588), (-0.759, 0, 0.588)]),
    "HO2": Atoms("OOH", positions=[(0, 0, 0), (0, -0.73, 1.264), (0.939, -0.8525, 1.4766)]),  
    "H2":  Atoms("HH",  positions=[(0, 0, 0), (0, 0, 0.74)]),
    "O2":  Atoms("OO",  positions=[(0, 0, 0), (0, 0, 1.21)]),
    "H":   Atoms("H",   positions=[(0, 0, 0)]),
    "O":   Atoms("O",   positions=[(0, 0, 0)])
}

CLOSED_SHELL_MOLECULES = ["H2", "H2O"]
SLAB_VACUUM, GAS_BOX, ADSORBATE_HEIGHT = 15.0, 15.0, 2.0


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
    
    molecule = set_initial_magmoms(molecule, kind="gas", formula=molecule_name)
    optimized_molecule = my_calculator(molecule, "gas", calculator=calculator, 
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
    bulk_structure = set_initial_magmoms(bulk_structure, kind="bulk")
    
    optimized_bulk = my_calculator(
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
    slab = set_initial_magmoms(slab, kind="slab")
    
    optimized_slab = my_calculator(
        slab, "slab",
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
        with open(result_file, 'r') as f:
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
        raise ValueError(f"Invalid position specification: {position_spec}")
    
    # Add adsorbate
    ads_and_slab = slab.copy()
    if height is None:
        height = 2.0
    
    add_adsorbate(ads_and_slab, adsorbate, height=height, position=position)
    set_initial_magmoms(ads_and_slab)
    
    ads_and_slab = my_calculator(ads_and_slab, "slab", calculator=calculator_type, 
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
    results = {}
    
    # Gas-phase molecules
    for molecule in ["O2", "H2", "H2O", "OH", "HO2", "O"]:
        mol_dir = base_dir / f"gas_{molecule}"
        if not overwrite and (mol_dir / "result.json").exists():
            with open(mol_dir / "result.json", 'r') as f:
                results[f"gas_{molecule}"] = json.load(f)["E_total"]
        else:
            _, energy = optimize_gas_molecule(molecule, GAS_BOX, str(mol_dir), calculator, yaml_path)
            results[f"gas_{molecule}"] = energy
    
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
            ads_dir = str(base_dir / f"ads_{adsorbate}_{i}")
            if not overwrite:
                with open(Path(ads_dir) / "result.json", "r") as f:
                    energy = json.load(f)["E_total"]
            else:
                energy = get_adsorption_energy(slab, MOLECULES[adsorbate], position, ads_dir, calculator, yaml_path)
            adsorption_energies.append(energy)
        results[f"ads_{adsorbate}"] = min(adsorption_energies)
    
    # Add slab energy
    results["slab"] = E_slab
    
    # Compute ORR reaction energies
    energies = {
        "O2": results["gas_O2"],
        "H2": results["gas_H2"],
        "H2O": results["gas_H2O"],
        "OOH": results["ads_OOH"],
        "O": results["ads_O"],
        "OH": results["ads_OH"],
        "slab": E_slab
    }
    
    # --- ORR reaction steps ---
    # 1) O2 + * + H+ + e- → OOH*
    # 2) OOH* + H+ + e- → O* + H2O
    # 3) O* + H+ + e- → OH*
    # 4) OH* + H+ + e- → * + H2O
    deltaE1 = energies["OOH"] - energies["slab"] - energies["O2"] - 0.5 * energies["H2"]
    deltaE2 = energies["O"] + energies["H2O"] - energies["OOH"] - 0.5 * energies["H2"]
    deltaE3 = energies["OH"] - energies["O"] - 0.5 * energies["H2"]
    deltaE4 = energies["slab"] + energies["H2O"] - energies["OH"] - 0.5 * energies["H2"]
    
    return [deltaE1, deltaE2, deltaE3, deltaE4], energies


def plot_free_energy_diagram(deltaGs: List[float], work_dir: Path):
    """Plot free energy diagram."""
    try:
        import matplotlib.pyplot as plt

        steps = ["O2 + * + H+ + e-", "OOH*", "O* + H2O", "OH*", "* + H2O"]
        cumulative_energies = [0] + np.cumsum(deltaGs).tolist()
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(steps)), cumulative_energies, 'o-', linewidth=2, markersize=8)
        plt.xlabel("Reaction Step")
        plt.ylabel("Free Energy (eV)")
        plt.title("ORR Free Energy Diagram")
        plt.xticks(range(len(steps)), steps, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(work_dir / "free_energy_diagram.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        logger = setup_logging()
        logger.warning("Matplotlib not available. Skipping plot generation.")


def get_orr_overpotential(bulk: Atoms = None, deltaEs: List[float] = None,
                          work_dir: Union[str, Path] = "result/matter_sim", 
                          overwrite: bool = False, log_level: str = "INFO", 
                          calculator: str = "mace", verbose: bool = True, 
                          save_plot: bool = False, adsorbates: Dict[str, List[Tuple[float, float]]] = None,
                          yaml_path: str = None) -> Dict[str, Any]:
    """Calculate ORR overpotential from bulk structure or reaction energies.
    
    Args:
        bulk: Bulk structure (required for full calculation)
        deltaEs: Pre-calculated reaction energies (alternative to bulk)
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

    U0 = 1.23  # equilibrium potential
    
    # Full calculation from bulk structure
    if bulk is None:
        raise ValueError("Either bulk structure or reaction_energies must be provided")
    
    logger.info("Starting ORR overpotential calculation from bulk structure...")
    
    # 1. Optimize bulk structure
    logger.info("Optimizing bulk structure...")
    bulk_dir = str(base_path / "bulk")
    optimized_bulk, bulk_energy = optimize_bulk(bulk, bulk_dir, calculator, yaml_path)
    
    # 2. Optimize slab structure
    logger.info("Optimizing slab...")
    slab_dir = str(base_path / "slab")
    optimized_slab, E_slab = optimize_slab(optimized_bulk, slab_dir, calculator, yaml_path)
    
    # 3. Calculate all required molecules and reaction energies
    logger.info("Calculating required molecules and reaction energies...")
    deltaEs, Es = get_deltaEs(optimized_slab, E_slab, base_path, calculator, yaml_path, adsorbates, overwrite)

    # Convert to free energies and calculate overpotential
    deltaGs = [deltaE + U0 for deltaE in deltaEs]
    overpotential = max(deltaGs) - U0
    
    if verbose:
        logger.info(f"Reaction energies: {deltaEs}")
        logger.info(f"Free energies: {deltaGs}")
        logger.info(f"Overpotential: {overpotential:.3f} V")
    
    if save_plot:
        plot_free_energy_diagram(deltaGs, base_path)
    
    # 5. Save summary
    summary_file = base_path / "ORR_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("--- ORR Summary ---\n\n")
        f.write(json.dumps(convert_numpy_types(Es), indent=2))
        f.write(f"\n\ndeltaE (eV): {', '.join(f'{e:+.3f}' for e in deltaEs)}\n")
        f.write(f"Overpotential η = {overpotential:.3f} V\n")
    
    logger.info(f"ORR calculation completed. Overpotential: {overpotential:.3f} V")
    return {
        "eta": overpotential,
        "reaction_energies": deltaEs,
        "free_energies": deltaGs
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
