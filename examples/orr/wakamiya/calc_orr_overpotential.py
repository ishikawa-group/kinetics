#!/usr/bin/env python3
"""
ORR Overpotential Workflow (Offset-based Adsorption Version)
============================================================

- Gas-phase optimization: O2, H2, H2O, OH, HO2(=OOH), O (6 molecules total)
- Adsorption optimization: OOH*, O*, OH* (3 adsorbates total)

The lowest energy configuration is adopted as the representative value for each
adsorbate species to evaluate ΔE and η.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.io import read, write

# External helper functions
from orr_overpotential_calculator.calc_orr_energy import (
    optimize_bulk_structure,
    optimize_slab_structure,
    optimize_gas_molecule,
    optimize_cluster_structure,
    calculate_adsorption_with_offset,
    calculate_adsorption_with_indices,
    attach_modifier_to_surface,
)
from .tool import convert_numpy_types

np.set_printoptions(precision=3)

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------

# Molecular library (gas-phase includes all species, adsorbates are subset)
MOLECULES: Dict[str, Atoms] = {
    # Adsorbates (gas + adsorption calculations)
    "OH": Atoms("OH", positions=[(0, 0, 0), (0, 0, 0.97)]),
    "HO2": Atoms("OOH", positions=[(0, 0, 0), (0, -0.73, 1.264), (0.939, -0.8525, 1.4766)]),
    "O": Atoms("O", positions=[(0, 0, 0)]),
    # Gas-phase only
    "O2": Atoms("OO", positions=[(0, 0, 0), (0, 0, 1.21)]),
    "H2O": Atoms("OHH", positions=[(0, 0, 0), (0.759, 0, 0.588), (-0.759, 0, 0.588)]),
    "H2": Atoms("HH", positions=[(0, 0, 0), (0, 0, 0.74)]),
}

# Molecules that skip adsorption calculations
GAS_ONLY: set[str] = {"H2", "O2", "H2O"}

# Default adsorption sites (fractional coordinates)
ADSORBATES: Dict[str, List[Tuple[float, float]]] = {
    "HO2": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],  # ontop, bridge, hollow
    "O": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],
    "OH": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],
}

# Structural parameters (in Angstroms)
SLAB_VACUUM = 15.0
GAS_BOX = 15.0
ADSORBATE_HEIGHT = 2.0

# Logger setup
logger = logging.getLogger("orr_workflow")

# ---------------------------------------------------------------------------
# Adsorption Energy Calculation Functions (Offset-based)
# ---------------------------------------------------------------------------


def calculate_required_molecules(
        optimized_slab: Atoms,
        slab_energy: float,
        outdir: Path,
        overwrite: bool = False,
        calculator: str = "mace",
        adsorbates: Dict[str, List[Tuple[float, float]]] = None,
        yaml_path: str = None,
    ) -> Dict[str, Any]:
    """
    Calculate gas-phase and adsorption energies for all required molecules.
    
    Args:
        optimized_slab: Optimized slab structure
        slab_energy: Energy of the optimized slab
        outdir: Base directory for calculations
        overwrite: Force recalculation of existing results
        calculator: Calculator type ("vasp", "mace")
        adsorbates: Dictionary of adsorbate positions (default uses ADSORBATES)
        yaml_path: Path to VASP configuration file
        
    Returns:
        Dictionary containing all calculation results
    """
    results: Dict[str, Any] = {}
    outdir.mkdir(parents=True, exist_ok=True)

    # Use default adsorbates if none provided
    if adsorbates is None:
        adsorbates = ADSORBATES

    for molecule_name, molecule in MOLECULES.items():
        logger.info("=== Processing %s ===", molecule_name)
        molecule_dir = outdir / molecule_name
        gas_dir = molecule_dir / f"{molecule_name}_gas"
        adsorption_dir = molecule_dir / "adsorption"
        gas_dir.mkdir(parents=True, exist_ok=True)
        adsorption_dir.mkdir(parents=True, exist_ok=True)

        # 1. Gas-phase optimization
        gas_json = gas_dir / "opt_result.json"
        xyz_gas = gas_dir / "opt.xyz"

        optimized_molecule, gas_energy = optimize_gas_molecule(
            molecule_name, GAS_BOX, str(gas_dir), calculator, yaml_path
        )
        optimized_molecule.write(xyz_gas)
        json.dump({"E_opt": float(gas_energy)}, gas_json.open("w"))

        results.setdefault(molecule_name, {})["E_gas"] = float(gas_energy)

        # 2. Skip adsorption for gas-only molecules
        if molecule_name in GAS_ONLY:
            continue

        # 3. Adsorption calculations at different offsets
        offsets = adsorbates.get(molecule_name, [])
        offset_data: Dict[str, Dict[str, float]] = {}

        for offset in offsets:
            key = f"ofst_{offset[0]}_{offset[1]}"
            offset_json = adsorption_dir / f"{key}.json"
            work_dir = adsorption_dir / key

            if offset_json.exists() and (work_dir / ".done").exists() and not overwrite:
                # Load existing results
                data = json.load(offset_json.open())
                total_energy = data["E_total"]
                elapsed_time = data["elapsed"]
            else:
                # Perform new calculation
                total_energy, elapsed_time = calculate_adsorption_with_offset(
                    optimized_slab, optimized_molecule, offset, str(work_dir),
                    calculator, yaml_path
                )
                json.dump({
                    "E_total": total_energy,
                    "elapsed": elapsed_time
                }, offset_json.open("w"))
                (work_dir / ".done").touch()

            offset_data[key] = {"E_total": total_energy, "elapsed": elapsed_time}

        # 4. Find configuration with lowest energy
        if offset_data:
            best_key, best_energy = min(
                ((k, d["E_total"]) for k, d in offset_data.items()),
                key=lambda x: x[1]
            )
            best_adsorption_energy = best_energy - (slab_energy + gas_energy)
            results[molecule_name].update({
                "E_slab": float(slab_energy),
                "E_total_best": float(best_energy),
                "best_offset": best_key,
                "E_ads_best": float(best_adsorption_energy),
                "offsets": offset_data,
            })
            logger.info("  -> Best offset: %s   E_ads = %.3f eV", best_key, best_adsorption_energy)

    # 5. Save summary results
    json.dump(
        convert_numpy_types(results),
        (outdir / "all_results.json").open("w"),
        indent=2
    )
    return results


def calculate_required_molecules_with_indices(
        optimized_slab: Atoms,
        slab_energy: float,
        outdir: Path,
        overwrite: bool = False,
        calculator: str = "mace",
        indices_dict: Dict[str, List] = None,
        yaml_path: str = None,
        height: float = None,
        orientation: list = None,
    ) -> Dict[str, Any]:
    """
    Calculate adsorption energies using specified atomic indices.
    
    Args:
        optimized_slab: Optimized slab structure
        slab_energy: Energy of the optimized slab
        outdir: Base directory for calculations
        overwrite: Force recalculation of existing results
        calculator: Calculator type
        indices_dict: Dictionary of atomic indices for each molecule
        yaml_path: Path to configuration file
        height: Adsorption height (optional)
        orientation: Molecular orientation vector (optional)
        
    Returns:
        Dictionary containing all calculation results
    """
    results: Dict[str, Any] = {}
    outdir.mkdir(parents=True, exist_ok=True)

    # Use default indices if none provided
    if indices_dict is None:
        indices_dict = {
            "HO2": [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],
            "O": [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],
            "OH": [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],
        }

    for molecule_name, molecule in MOLECULES.items():
        logger.info("=== Processing %s ===", molecule_name)
        molecule_dir = outdir / molecule_name
        gas_dir = molecule_dir / f"{molecule_name}_gas"
        adsorption_dir = molecule_dir / "adsorption"
        gas_dir.mkdir(parents=True, exist_ok=True)
        adsorption_dir.mkdir(parents=True, exist_ok=True)

        # 1. Gas-phase optimization
        gas_json = gas_dir / "opt_result.json"
        xyz_gas = gas_dir / "opt.xyz"

        optimized_molecule, gas_energy = optimize_gas_molecule(
            molecule_name, GAS_BOX, str(gas_dir), calculator, yaml_path
        )
        optimized_molecule.write(xyz_gas)
        json.dump({"E_opt": float(gas_energy)}, gas_json.open("w"))

        results.setdefault(molecule_name, {})["E_gas"] = float(gas_energy)

        # 2. Skip adsorption for gas-only molecules
        if molecule_name in GAS_ONLY:
            continue

        # 3. Adsorption calculations at specified indices
        indices_list = indices_dict.get(molecule_name, [])
        indices_data: Dict[str, Dict[str, float]] = {}

        for indices in indices_list:
            # Create key from indices
            indices_str = "_".join(map(str, indices))
            key = f"idx_{indices_str}"
            indices_json = adsorption_dir / f"{key}.json"
            work_dir = adsorption_dir / key

            if indices_json.exists() and (work_dir / ".done").exists() and not overwrite:
                # Load existing results
                data = json.load(indices_json.open())
                total_energy = data["E_total"]
                elapsed_time = data["elapsed"]
            else:
                # Perform new calculation using indices
                total_energy, elapsed_time = calculate_adsorption_with_indices(
                    optimized_slab, optimized_molecule, indices, str(work_dir),
                    height=height, orientation=orientation,
                    calculator=calculator, yaml_path=yaml_path
                )
                json.dump({
                    "E_total": total_energy,
                    "elapsed": elapsed_time
                }, indices_json.open("w"))
                (work_dir / ".done").touch()

            indices_data[key] = {"E_total": total_energy, "elapsed": elapsed_time}

        # 4. Find configuration with lowest energy
        if indices_data:
            best_key, best_energy = min(
                ((k, d["E_total"]) for k, d in indices_data.items()),
                key=lambda x: x[1]
            )
            best_adsorption_energy = best_energy - (slab_energy + gas_energy)
            results[molecule_name].update({
                "E_slab": float(slab_energy),
                "E_total_best": float(best_energy),
                "best_site": best_key,
                "E_ads_best": float(best_adsorption_energy),
                "sites": indices_data,
            })
            logger.info("  -> Best site: %s   E_ads = %.3f eV",
                        best_key, best_adsorption_energy)

    # 5. Save summary results
    json.dump(
        convert_numpy_types(results),
        (outdir / "all_results.json").open("w"),
        indent=2
    )
    return results


# ---------------------------------------------------------------------------
# Reaction Energy and Overpotential Calculation Functions
# ---------------------------------------------------------------------------

def compute_reaction_energies(
        results: Dict[str, Any],
        slab_energy: float
    ) -> Tuple[List[float], Dict[str, float]]:
    """
    Compute reaction energies for the 4-electron ORR pathway.
    
    Args:
        results: Dictionary containing calculation results
        slab_energy: Energy of the clean slab
        
    Returns:
        Tuple of (reaction energies list, energies dictionary)
    """
    def get_gas_energy(molecule: str) -> float:
        return results[molecule]["E_gas"]

    def get_total_energy(molecule: str) -> float:
        return results[molecule]["E_total_best"]

    # Gas-phase energies
    E_H2_gas = get_gas_energy("H2")
    E_H2O_gas = get_gas_energy("H2O")
    # O2(g) energy corrected (SI of Bligaard/Nørskov)
    E_O2_gas = 2 * (2.46 + E_H2O_gas - E_H2_gas)

    # Slab+adsorbate total energies
    E_slab_OOH = get_total_energy("HO2")  # HO2 = OOH*
    E_slab_O = get_total_energy("O")
    E_slab_OH = get_total_energy("OH")

    # Apply solvent corrections
    # References: https://doi.org/10.1016/j.cattod.2018.07.036, https://doi.org/10.1039/D0NR03339A
    E_slab_OOH = E_slab_OOH - 0.1
    E_slab_OH = E_slab_OH - 0.2

    # Store all energies
    energies = {
        "E_H2_g": E_H2_gas,
        "E_H2O_g": E_H2O_gas,
        "E_O2_g": E_O2_gas,
        "E_slab": slab_energy,
        "E_slab_OOH": E_slab_OOH,
        "E_slab_O": E_slab_O,
        "E_slab_OH": E_slab_OH,
    }

    # Calculate reaction energies ΔE for 4-electron ORR pathway
    dE1 = E_slab_OOH - (E_O2_gas + slab_energy + 0.5 * E_H2_gas)  # O2(g) + * + ½H2 → OOH*
    dE2 = (E_slab_O + E_H2O_gas) - (E_slab_OOH + 0.5 * E_H2_gas)  # OOH* + ½H2 → O* + H2O
    dE3 = E_slab_OH - (E_slab_O + 0.5 * E_H2_gas)  # O* + ½H2 → OH*
    dE4 = (slab_energy + E_H2O_gas) - (E_slab_OH + 0.5 * E_H2_gas)  # OH* + ½H2 → * + H2O

    reaction_energies = [dE1, dE2, dE3, dE4]
    energies.update({
        "dE1": dE1, "dE2": dE2, "dE3": dE3, "dE4": dE4,
    })

    return reaction_energies, energies


def get_overpotential_orr(
        reaction_energies: List[float],
        output_dir: Path,
        temperature: float = 298.15,
        verbose: bool = False,
        save_plot: bool = True,
    ) -> Dict[str, Any]:
    """
    Calculate ORR overpotential and generate free-energy diagram.
    
    Args:
        reaction_energies: List of 4 reaction energies (eV)
        output_dir: Directory for output files
        temperature: Temperature in Kelvin
        verbose: Print detailed information
        save_plot: Whether to save the free-energy diagram plot
        
    Returns:
        Dictionary containing overpotential and thermodynamic data
    """
    reaction_count = 4  # 4-electron pathway
    assert len(reaction_energies) == reaction_count, "reaction_energies must contain 4 elements"

    # Zero-point energy corrections (eV)-- Reference: https://doi.org/10.1021/ja405997s
    zpe = {
        "H2": 0.35, "H2O": 0.57,
        "Oads": 0.06, "OHads": 0.37, "OOHads": 0.44,
    }

    # Entropy terms T*S (eV) -- Reference: https://doi.org/10.1021/ja405997s
    #
    entropy = {
        "H2": 0.403 / temperature, "H2O": 0.67 / temperature,
        "Oads": 0.0, "OHads": 0.0, "OOHads": 0.0,
    }

    # Calculate O2 corrections
    zpe["O2"] = 0 + 2 * (zpe["H2O"] - zpe["H2"])
    entropy["O2"] = 0 + 2 * (entropy["H2O"] - entropy["H2"])

    # Calculate ZPE and entropy corrections for each reaction step
    delta_zpe = np.array([
        zpe["OOHads"] + (-0.5 * zpe["H2"] + -zpe["O2"]),
        zpe["Oads"] + zpe["H2O"] - zpe["OOHads"] - 0.5 * zpe["H2"],
        zpe["OHads"] - zpe["Oads"] - 0.5 * zpe["H2"],
        zpe["H2O"] - zpe["OHads"] - 0.5 * zpe["H2"],
    ])

    delta_ts = np.array([
        temperature * entropy["OOHads"] + (-0.5 * temperature * entropy["H2"] + -temperature * entropy["O2"]),
        temperature * entropy["Oads"] + temperature * entropy["H2O"] - temperature * entropy[
            "OOHads"] - 0.5 * temperature * entropy["H2"],
        temperature * entropy["OHads"] - temperature * entropy["Oads"] - 0.5 * temperature * entropy["H2"],
        temperature * entropy["H2O"] - temperature * entropy["OHads"] - 0.5 * temperature * entropy["H2"],
    ])

    # Calculate free energies
    reaction_energies = np.array(reaction_energies)
    delta_g_u0 = reaction_energies + delta_zpe - delta_ts  # ΔG at U=0 V

    # Free energy profiles
    g_profile_u0 = np.concatenate(([0.0], np.cumsum(delta_g_u0)))
    equilibrium_potential = 1.23  # V

    # Calculate step-wise free energy changes
    diff_g_u0 = np.diff(g_profile_u0)

    # Find limiting potential and overpotential
    dg_orr_max = np.max(diff_g_u0)
    limiting_potential = (-1) * dg_orr_max
    overpotential = equilibrium_potential - limiting_potential

    # Calculate profiles for U=1.23V and U=limiting potential
    g_profile_ueq = g_profile_u0 - np.arange(reaction_count + 1) * (-1) * equilibrium_potential
    g_profile_ul = g_profile_u0 - np.arange(reaction_count + 1) * (-1) * limiting_potential
    diff_g_eq = np.diff(g_profile_ueq)
    diff_g_ul = np.diff(g_profile_ul)

    # Generate free-energy diagram plot
    if save_plot:
        import matplotlib.pyplot as plt

        # Reaction step labels
        labels = [
            "O$_2$ + 2H$_2$", "OOH* + 1.5H$_2$", "O* + H$_2$O + H$_2$",
            "OH* + H$_2$O + 0.5H$_2$", "* + 2H$_2$O",
        ]

        # Steps and relative profiles
        steps = np.arange(reaction_count + 1)
        g0_shift = g_profile_u0 - g_profile_u0[-1]
        geq_shift = g_profile_ueq - g_profile_ueq[-1]
        gul_shift = g_profile_ul - g_profile_ul[-1]

        # Colors for different potential profiles
        u0_color = 'black'  # U=0V is black
        ueq_color = 'green'  # U=1.23V is green
        ul_color = 'blue'  # U=UL is blue

        # Horizontal line width
        line_width = 0.3

        # Create figure
        plt.figure(figsize=(8, 7))

        # ------ U=0V profile ------
        # First point with label
        plt.hlines(g0_shift[0], steps[0] - line_width, steps[0] + line_width,
                   color=u0_color, alpha=0.6, linewidth=2.5, label="U = 0 V")

        # Remaining points without label
        for i in range(1, len(steps)):
            plt.hlines(g0_shift[i], steps[i] - line_width, steps[i] + line_width,
                       color=u0_color, alpha=0.6, linewidth=2.5)

        # Connect points with dashed lines
        for i in range(len(steps) - 1):
            plt.plot([steps[i] + line_width, steps[i + 1] - line_width],
                     [g0_shift[i], g0_shift[i + 1]],
                     '--', color=u0_color, alpha=0.6, linewidth=1.0)

        # Add markers
        plt.plot(steps, g0_shift, 'o', color=u0_color, alpha=0.6,
                 markersize=4, linestyle='none')

        # ------ U=UL (Limiting Potential) profile ------
        # First point with label
        plt.hlines(gul_shift[0], steps[0] - line_width, steps[0] + line_width,
                   color=ul_color, linewidth=2.5,
                   label=f"U$_{{L}}$ = {limiting_potential:.2f} V")

        # Remaining points without label
        for i in range(1, len(steps)):
            plt.hlines(gul_shift[i], steps[i] - line_width, steps[i] + line_width,
                       color=ul_color, linewidth=2.5)

        # Connect points with dashed lines
        for i in range(len(steps) - 1):
            plt.plot([steps[i] + line_width, steps[i + 1] - line_width],
                     [gul_shift[i], gul_shift[i + 1]],
                     '--', color=ul_color, linewidth=1.0)

        # Add markers
        plt.plot(steps, gul_shift, 's', color=ul_color, markersize=5, linestyle='none')

        # ------ U=Equilibrium (1.23V) profile ------
        # First point with label
        plt.hlines(geq_shift[0], steps[0] - line_width, steps[0] + line_width,
                   color=ueq_color, alpha=0.8, linewidth=2.5,
                   label=f"U = {equilibrium_potential} V")

        # Remaining points without label
        for i in range(1, len(steps)):
            plt.hlines(geq_shift[i], steps[i] - line_width, steps[i] + line_width,
                       color=ueq_color, alpha=0.8, linewidth=2.5)

        # Connect points with dashed lines
        for i in range(len(steps) - 1):
            plt.plot([steps[i] + line_width, steps[i + 1] - line_width],
                     [geq_shift[i], geq_shift[i + 1]],
                     '--', color=ueq_color, alpha=0.8, linewidth=1.0)

        # Add markers
        plt.plot(steps, geq_shift, 'o', color=ueq_color, alpha=0.8,
                 markersize=6, linestyle='none')

        # Formatting
        plt.xticks(steps, labels, rotation=15, ha='right')
        plt.ylabel("ΔG (eV)", fontsize=12, fontweight='bold')
        plt.xlabel("Reaction Coordinate", fontsize=12, fontweight='bold')
        plt.title("4e⁻ ORR Free-Energy Diagram", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.3)

        # Add legend and horizontal zero line
        plt.legend(loc='upper right', fontsize=10)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)

        plt.tight_layout()

        # Save figure
        figure_path = output_dir / "ORR_free_energy_diagram.png"
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Saved diagram → %s", figure_path)
    else:
        logger.info("Plot generation skipped (save_plot=False)")

    if verbose:
        logger.info("ΔG (U=0) = %s", delta_g_u0)
        logger.info("Limiting potential U_L = %.3f V", limiting_potential)
        logger.info("Overpotential η = %.3f V", overpotential)

    return {
        "eta": overpotential,
        "diffG_U0": diff_g_u0.tolist(),
        "diffG_eq": diff_g_eq.tolist(),
        "U_L": limiting_potential,
        "G_profile_U0": g_profile_u0.tolist(),
        "G_profile_Ueq": g_profile_ueq.tolist(),
        "G_profile_UL": g_profile_ul.tolist()
    }


# ---------------------------------------------------------------------------
# Main Workflow Functions
# ---------------------------------------------------------------------------

def calc_orr_overpotential(
        bulk: Atoms,
        outdir: str = "result/matter_sim",
        overwrite: bool = False,
        log_level: str = "INFO",
        calculator: str = "mace",
        adsorbates: Dict[str, List[Tuple[float, float]]] = None,
        yaml_path: str = None,
    ) -> Dict[str, Any]:
    """
    Calculate ORR overpotential for slab systems.
    
    Args:
        bulk: Bulk crystal structure
        outdir: Output directory for calculations
        overwrite: Force recalculation of existing results
        log_level: Logging level
        calculator: Calculator type ("vasp", "mace")
        adsorbates: Dictionary of adsorption sites
        yaml_path: Path to VASP configuration file

    Returns:
        Dictionary containing overpotential and thermodynamic data
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    # Use global default if adsorbates not provided
    if adsorbates is None:
        adsorbates = ADSORBATES

    outdir_path = Path(outdir).resolve()
    outdir_path.mkdir(parents=True, exist_ok=True)

    # vaspでない場合はbulkディレクトリを作成
    if calculator != "vasp":
        bulk_dir = outdir_path / "bulk"
        bulk_dir.mkdir(parents=True, exist_ok=True)

    # vaspでない場合はslabディレクトリを作成
    if calculator != "vasp":
        slab_dir = outdir_path / "slab"
        slab_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bulk optimization
    logger.info("Optimizing bulk structure...")
    optimized_bulk, bulk_energy = optimize_bulk_structure(
        bulk, str(outdir_path / "bulk"), calculator, yaml_path
    )
    write(str(outdir_path / "bulk" / "optimized_bulk.xyz"), optimized_bulk)

    # 2. Clean slab optimization
    logger.info("Optimizing clean slab...")
    optimized_slab, slab_energy = optimize_slab_structure(
        optimized_bulk, str(outdir_path / "slab"), calculator, yaml_path
    )
    write(str(outdir_path / "slab" / "optimized_slab.xyz"), optimized_slab)

    # 3. Gas and adsorption calculations (offset scheme)
    logger.info("Running required molecule calculations...")
    results = calculate_required_molecules(
        optimized_slab, slab_energy, outdir_path,
        overwrite=overwrite, calculator=calculator, adsorbates=adsorbates, yaml_path=yaml_path,
    )

    # 4. Calculate reaction energies and overpotential
    reaction_energies, energies = compute_reaction_energies(results, slab_energy)
    orr_results = get_overpotential_orr(reaction_energies, outdir_path, verbose=True, save_plot=True)
    overpotential = orr_results["eta"]

    # 5. Write summary
    with (outdir_path / "ORR_summary.txt").open("w") as f:
        f.write("--- ORR Summary ---\n\n")
        f.write(json.dumps(convert_numpy_types(energies), indent=2))
        f.write("\n\nΔE (eV): " + ", ".join(f"{e:+.3f}" for e in reaction_energies) + "\n")
        f.write(f"Overpotential η = {overpotential:.3f} V\n")
    logger.info("Summary written → %s", outdir_path / "ORR_summary.txt")

    return orr_results


def calc_cluster_orr_overpotential(
        cluster: Atoms,
        outdir: str = "result/matter_sim",
        overwrite: bool = False,
        log_level: str = "INFO",
        calculator: str = "mace",
        adsorbates: Dict[str, List[Tuple]] = None,
        yaml_path: str = None,
        vacuum_size: float = 20.0,
    ) -> Dict[str, Any]:
    """
    Calculate ORR overpotential for cluster systems.
    
    Args:
        cluster: cluster structure
        outdir: Base directory for calculations
        overwrite: Force recalculation of existing results
        log_level: Logging level
        calculator: Calculator type ("vasp", "mace")
        adsorbates: Dictionary of atomic indices for adsorption sites
        yaml_path: Path to configuration file
        vacuum_size: Vacuum size around cluster (Å)
        
    Returns:
        Dictionary containing overpotential and thermodynamic data
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    outdir_path = Path(outdir).resolve()
    outdir_path.mkdir(parents=True, exist_ok=True)

    # vaspでない場合はclusterディレクトリを作成
    if calculator != "vasp":
        cluster_dir = outdir_path / "cluster"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # 1. cluster optimization
    logger.info("Optimizing cluster...")

    # Calculate cluster dimensions
    positions = cluster.get_positions()
    x_size = positions[:, 0].max() - positions[:, 0].min()
    y_size = positions[:, 1].max() - positions[:, 1].min()
    z_size = positions[:, 2].max() - positions[:, 2].min()
    cluster_diameter = max(x_size, y_size, z_size)
    gas_box = cluster_diameter + vacuum_size

    optimized_cluster, cluster_energy = optimize_cluster_structure(
        cluster, gas_box, str(outdir_path / "cluster"), calculator, yaml_path
    )

    write(str(outdir_path / "cluster" / "optimized_cluster.xyz"), optimized_cluster)

    # 2. Gas and adsorption calculations (index scheme)
    logger.info("Running required molecule calculations...")

    # Convert adsorbates to indices_dict
    indices_dict = None
    if adsorbates:
        indices_dict = adsorbates
    else:
        # Default index dictionary
        indices_dict = {
            "HO2": [(0,)],
            "O": [(0,)],
            "OH": [(0,)],
        }

    results = calculate_required_molecules_with_indices(
        optimized_cluster, cluster_energy, outdir_path,
        overwrite=overwrite, calculator=calculator, indices_dict=indices_dict, yaml_path=yaml_path,
    )

    # 3. Calculate reaction energies and overpotential
    reaction_energies, energies = compute_reaction_energies(results, cluster_energy)
    orr_results = get_overpotential_orr(reaction_energies, outdir_path, verbose=True, save_plot=True)

    # 4. Write summary
    with (outdir_path / "ORR_summary.txt").open("w") as f:
        f.write("--- ORR Summary ---\n\n")
        f.write(json.dumps(convert_numpy_types(energies), indent=2))
        f.write("\n\nΔE (eV): " + ", ".join(f"{e:+.3f}" for e in reaction_energies) + "\n")
        f.write(f"Overpotential η = {orr_results['eta']:.3f} V\n")
    logger.info("Summary written → %s", outdir_path / "ORR_summary.txt")

    return orr_results


def calc_orr_overpotential_modified(
    bulk: Atoms,
    base_dir: str = "result/modified_surface",
    overwrite: bool = False,
    log_level: str = "INFO",
    calculator: str = "mace",
    orr_adsorbates: Dict[str, List[Tuple[float, float]]] = None,
    modify_adsorbates: Dict[str, Atoms] = None,
    modify_offset: Dict[str, List[Tuple[float, float]]] = None,
    yaml_path: str = None,
) -> Dict[str, Any]:
    """
    Calculate ORR overpotential on surface modified with adsorbates.

    Args:
        bulk: Bulk crystal structure
        base_dir: Base directory for calculation results
        overwrite: Force recalculation of existing results
        log_level: Logging level
        calculator: Calculator type ("vasp", "mace")
        orr_adsorbates: Adsorption sites for ORR-related species
        modify_adsorbates: Dictionary of modifier molecules {name: Atoms}
        modify_offset: Adsorption sites for modifier molecules {molecule_name: [(x,y)]}
        yaml_path: Path to VASP configuration file

    Returns:
        Dictionary containing overpotential and thermodynamic data
    """

    logger = logging.getLogger("orr_modified_surface")

    # Logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    # Set default values
    if orr_adsorbates is None:
        orr_adsorbates = ADSORBATES

    # Check modifier molecules and positions
    if modify_adsorbates is None or modify_offset is None:
        raise ValueError("Surface modifier molecules (modify_adsorbates) and adsorption positions (modify_offset) are required")

    # Directory setup
    base_path = Path(base_dir).resolve()
    base_path.mkdir(parents=True, exist_ok=True)

    # --- 1. Bulk optimization ---
    logger.info("Optimizing bulk structure...")
    bulk_dir = base_path / "bulk"
    bulk_dir.mkdir(parents=True, exist_ok=True)

    optimized_bulk, bulk_energy = optimize_bulk_structure(
        bulk, str(bulk_dir), calculator, yaml_path
    )
    write(str(bulk_dir / "optimized_bulk.xyz"), optimized_bulk)

    # --- 2. Clean slab optimization ---
    logger.info("Optimizing clean slab structure...")
    slab_dir = base_path / "slab"
    slab_dir.mkdir(parents=True, exist_ok=True)

    optimized_slab, slab_energy = optimize_slab_structure(
        optimized_bulk, str(slab_dir), calculator, yaml_path
    )
    write(str(slab_dir / "optimized_slab.xyz"), optimized_slab)

    # --- 3. Modifier molecule optimization and adsorption ---
    # Use the first modifier molecule
    modifier_name = list(modify_adsorbates.keys())[0]
    modifier_molecule = modify_adsorbates[modifier_name]
    modifier_offset = modify_offset[modifier_name][0]  # Use single position

    logger.info(f"Attaching surface modifier {modifier_name} at position {modifier_offset}...")

    modified_slab, modified_slab_energy = attach_modifier_to_surface(
        optimized_slab,
        slab_energy,
        modifier_name,
        modifier_molecule,
        modifier_offset,
        base_path,
        overwrite=overwrite,
        calculator=calculator,
        yaml_path=yaml_path
    )

    # Save modified slab
    modified_slab_path = base_path / f"modified_slab_{modifier_name}.xyz"
    modified_slab.write(str(modified_slab_path))
    logger.info(f"Saved modified slab structure: {modified_slab_path}")

    # --- 4. ORR-related molecule adsorption calculations (on modified surface) ---
    logger.info("Running ORR-related molecule calculations on modified surface...")
    result_dir = base_path / "orr_on_modified_surface"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Perform ORR molecule adsorption calculations on modified slab
    results = calculate_required_molecules(
        modified_slab,                # Modified slab
        modified_slab_energy,         # Modified slab energy
        result_dir,
        overwrite=overwrite,
        calculator=calculator,
        adsorbates=orr_adsorbates,
        yaml_path=yaml_path
    )

    # --- 5. Reaction energy and overpotential calculation ---
    reaction_energies, energies = compute_reaction_energies(results, modified_slab_energy)
    orr_results = get_overpotential_orr(
        reaction_energies, result_dir, verbose=True, save_plot=True
    )
    overpotential = orr_results["eta"]

    # --- 6. Summary generation ---
    with (base_path / "ORR_summary_modified_surface.txt").open("w") as f:
        f.write(f"--- ORR Summary (Surface modifier: {modifier_name}) ---\n\n")
        f.write(json.dumps(convert_numpy_types(energies), indent=2))
        f.write("\n\nΔE (eV): " + ", ".join(f"{e:+.3f}" for e in reaction_energies) + "\n")
        f.write(f"Overpotential η = {overpotential:.3f} V\n")

    logger.info(f"Saved summary: {base_path / 'ORR_summary_modified_surface.txt'}")

    # Return results including modifier information
    orr_results.update({
        "modifier": modifier_name,
        "modifier_offset": modifier_offset,
    })

    return orr_results
