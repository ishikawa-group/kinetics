#!/usr/bin/env python3
"""
Calculate energies of the ORR reaction
"""

import os
import sys
import json
import time
import logging

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

from ase import Atom, Atoms
from ase.build import fcc111, bulk, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.vasp import Vasp
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter, ExpCellFilter
from ase.io import write

# Add custom module path
from orr_overpotential_calculator.tool import (
    parallel_displacement,
    fix_lower_surface,
    set_initial_magmoms,
    my_calculator,
    convert_numpy_types,
    place_adsorbate
    )

# ----------------------------------------------------------------------
# Configuration Constants
# ----------------------------------------------------------------------      

# Base directory for calculation results
BASE_DIR = "result"

# VASP configuration file path
YAML_PATH = os.path.join(Path(__file__).parent, "data", "vasp.yaml")

# Available adsorption sites
SITES = ["ontop", "bridge", "fcc"] 

# Molecular geometries (all positions in Angstroms)
MOLECULES = {
    "OH":  Atoms("OH",  positions=[(0, 0, 0), (0, 0, 0.97)]),
    "H2O": Atoms("OHH", positions=[(0, 0, 0), (0.759, 0, 0.588), (-0.759, 0, 0.588)]),
    "HO2": Atoms("OOH", positions=[(0, 0, 0), (0, -0.73, 1.264), (0.939, -0.8525, 1.4766)]),  
    "H2":  Atoms("HH",  positions=[(0, 0, 0), (0, 0, 0.74)]),
    "O2":  Atoms("OO",  positions=[(0, 0, 0), (0, 0, 1.21)]),
    "H":   Atoms("H",   positions=[(0, 0, 0)]),
    "O":   Atoms("O",   positions=[(0, 0, 0)])
    }

# Closed-shell molecules list for spin calculations
CLOSED_SHELL_MOLECULES = ["H2", "H2O"]

# Structural parameters (all in Angstroms)
SLAB_VACUUM = 15.0      # Vacuum layer thickness for slab calculations
GAS_BOX = 15.0          # Box size for gas phase calculations
ADSORBATE_HEIGHT = 2.0  # Initial height of adsorbate above surface

# ----------------------------------------------------------------------
# Optimization Functions
# ----------------------------------------------------------------------


def optimize_gas_molecule(
    molecule_name: str,
    gas_box_size: float,
    work_directory: str,
    calculator: str = "mace",
    yaml_path: str = YAML_PATH
    ) -> Tuple[Atoms, float]:
    """
    Optimize gas phase molecule structure and calculate energy.
    
    Args:
        molecule_name: Name of molecule from MOLECULES dictionary
        gas_box_size: Size of cubic simulation box (Å)
        work_directory: Directory for calculation files
        calculator: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Tuple of optimized Atoms object and energy (eV)
    """
    molecule = MOLECULES[molecule_name].copy()
    molecule.set_cell([gas_box_size, gas_box_size, gas_box_size])
    molecule.set_pbc(True)
    molecule.center()
    
    # Set initial magnetic moments
    molecule = set_initial_magmoms(molecule, kind="gas", formula=molecule_name)
    
    # Set up calculator and optimize
    optimized_molecule = my_calculator(
        molecule, "gas", 
        calculator=calculator,
        yaml_path=yaml_path, 
        calc_directory=work_directory
    )
    
    # For closed-shell molecules, set spin-unpolarized calculation
    if molecule_name in CLOSED_SHELL_MOLECULES:
        calculator = optimized_molecule.calc
        calculator.set(ispin=1)
        optimized_molecule.set_calculator(calculator)
    
    energy = optimized_molecule.get_potential_energy()
    return optimized_molecule, energy


def optimize_bulk_structure(
    bulk_atoms: Atoms,
    work_directory: str,
    calculator: str = "mace",
    yaml_path: str = YAML_PATH
    ) -> Tuple[Atoms, float]:
    """
    Optimize bulk crystal structure and calculate energy.
    
    Args:
        bulk_atoms: Initial bulk structure
        work_directory: Directory for calculation files
        calculator: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Tuple of optimized Atoms object and energy (eV)
    """
    bulk_structure = bulk_atoms.copy()
    bulk_structure.set_pbc(True)
    bulk_structure = set_initial_magmoms(bulk_structure, kind="bulk")
    
    optimized_bulk = my_calculator(
        bulk_structure, "bulk",
        calculator=calculator,
        yaml_path=yaml_path,
        calc_directory=work_directory
    )
    
    energy = optimized_bulk.get_potential_energy()
    return optimized_bulk, energy


def optimize_slab_structure(
    optimized_bulk: Atoms,
    work_directory: str,
    calculator: str = "mace",
    yaml_path: str = YAML_PATH
    ) -> Tuple[Atoms, float]:
    """
    Create and optimize slab structure from bulk and calculate energy.
    
    Args:
        optimized_bulk: Optimized bulk structure
        work_directory: Directory for calculation files
        calculator: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Tuple of optimized slab Atoms object and energy (eV)
    """
    slab = optimized_bulk.copy()
    slab.set_pbc(True)
    slab = parallel_displacement(slab, vacuum=SLAB_VACUUM)
    slab = fix_lower_surface(slab)
    slab = set_initial_magmoms(slab, kind="slab")
    
    optimized_slab = my_calculator(
        slab, "slab",
        calculator=calculator,
        yaml_path=yaml_path,
        calc_directory=work_directory
    )
    
    energy = optimized_slab.get_potential_energy()
    return optimized_slab, energy


def optimize_cluster_structure(
    cluster: Atoms,
    gas_box_size: float,
    work_directory: str,
    calculator: str = "mace",
    yaml_path: str = YAML_PATH
    ) -> Tuple[Atoms, float]:
    """
    Optimize cluster structure and calculate energy.
    
    Args:
        cluster: Initial cluster structure
        gas_box_size: Size of cubic simulation box (Å)
        work_directory: Directory for calculation files
        calculator: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Tuple of optimized Atoms object and energy (eV)
    """
    cluster_atoms = cluster.copy()
    cluster_atoms.set_cell([gas_box_size, gas_box_size, gas_box_size])
    cluster_atoms.set_pbc(True)
    cluster_atoms.center()
    cluster_atoms = set_initial_magmoms(cluster_atoms, kind="cluster")
    
    optimized_cluster = my_calculator(
        cluster_atoms, "cluster",
        calculator=calculator,
        yaml_path=yaml_path,
        calc_directory=work_directory
    )
    
    energy = optimized_cluster.get_potential_energy()
    return optimized_cluster, energy


def optimize_cluster_with_gas(
    cluster_gas: Atoms,
    work_directory: str,
    calculator: str = "mace",
    yaml_path: str = YAML_PATH
    ) -> Tuple[Atoms, float]:
    """
    Optimize cluster with adsorbed gas molecules and calculate energy.
    
    Args:
        cluster_gas: cluster with gas molecules
        work_directory: Directory for calculation files
        calculator: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Tuple of optimized Atoms object and energy (eV)
    """
    cluster_gas_atoms = cluster_gas.copy()
    cluster_gas_atoms.set_pbc(True)
    cluster_gas_atoms = set_initial_magmoms(
        cluster_gas_atoms, kind="cluster_gas"
    )
    
    optimized_cluster_gas = my_calculator(
        cluster_gas_atoms, "cluster_gas",
        calculator=calculator,
        yaml_path=yaml_path,
        calc_directory=work_directory
    )
    
    energy = optimized_cluster_gas.get_potential_energy()
    return optimized_cluster_gas, energy


# ----------------------------------------------------------------------
# Adsorption Calculation Functions
# ----------------------------------------------------------------------

def calculate_adsorption_on_site(
    optimized_slab: Atoms,
    optimized_molecule: Atoms,
    site: str,
    work_directory: str,
    calculator: str = "mace",
    yaml_path: str = YAML_PATH
    ) -> Tuple[float, float]:
    """
    Calculate adsorption energy at specified site.
    
    Args:
        optimized_slab: Optimized slab structure
        optimized_molecule: Optimized gas molecule structure
        site: Adsorption site type ("ontop", "bridge", "fcc")
        work_directory: Directory for calculation files
        calculator: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Tuple of total energy (eV) and calculation time (s)
    """
    print(f"   Site: {site}")
    site_directory = os.path.join(work_directory, f"site_{site}")
    os.makedirs(site_directory, exist_ok=True)

    # Record start time
    start_time = time.time()

    # Copy and prepare slab
    slab_atoms = optimized_slab.copy()
    slab_atoms = set_initial_magmoms(slab_atoms, kind="slab")

    # Copy and prepare adsorbate molecule
    adsorbate = optimized_molecule.copy()
    adsorbate = set_initial_magmoms(
        adsorbate, kind="gas", 
        formula=adsorbate.get_chemical_formula()
    )
    adsorbate.center()
    adsorbate.set_pbc(False)
    adsorbate.set_cell(None)

    # Create slab+adsorbate system
    slab_with_adsorbate = slab_atoms.copy()
    slab_with_adsorbate.set_pbc(True)
    slab_with_adsorbate = fix_lower_surface(slab_with_adsorbate)
    add_adsorbate(slab_with_adsorbate, adsorbate, height=ADSORBATE_HEIGHT, position=site)

    # Set up calculator and calculate energy
    slab_ads_calc = my_calculator(
        slab_with_adsorbate, "slab",
        calculator=calculator,
        yaml_path=yaml_path,
        calc_directory=site_directory
    )

    total_energy = slab_ads_calc.get_potential_energy()

    # Record elapsed time
    elapsed_time = time.time() - start_time

    # Save results
    output_file = os.path.join(site_directory, f"opt_slab_ads_{site}.xyz")
    write(output_file, slab_ads_calc)
    print(f"     E_total = {total_energy:.6f} eV, Time = {elapsed_time:.2f} s")

    return total_energy, elapsed_time


def calculate_adsorption_with_offset(
    optimized_slab: Atoms,
    optimized_molecule: Atoms,
    offset: Tuple[float, float],
    work_directory: str,
    calculator: str = "mace",
    yaml_path: str = YAML_PATH,
    ) -> Tuple[float, float]:
    """
    Calculate adsorption energy with specified offset position.
    
    Args:
        optimized_slab: Optimized slab structure
        optimized_molecule: Optimized gas molecule structure
        offset: Fractional coordinate offset (x, y)
        work_directory: Directory for calculation files
        calculator: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Tuple of total energy (eV) and calculation time (s)
    """
    os.makedirs(work_directory, exist_ok=True)
    start_time = time.time()

    # Prepare slab structure
    slab = optimized_slab.copy()
    slab = set_initial_magmoms(slab, kind="slab")
    slab.set_pbc(True)

    # Prepare adsorbate molecule
    adsorbate = optimized_molecule.copy()
    adsorbate = set_initial_magmoms(
        adsorbate,
        kind="gas",
        formula=adsorbate.get_chemical_formula()
    )
    adsorbate.center()
    adsorbate.set_pbc(False)
    adsorbate.set_cell(None)

    # Create slab+adsorbate system
    slab_with_adsorbate = slab.copy()
    slab_with_adsorbate = fix_lower_surface(slab_with_adsorbate)
    add_adsorbate(
        slab_with_adsorbate, adsorbate,
        ADSORBATE_HEIGHT,
        position="ontop",
        offset=offset
    )

    # Set up calculator and calculate energy
    calculator = my_calculator(
        slab_with_adsorbate,
        "slab",
        calculator=calculator,
        yaml_path=yaml_path,
        calc_directory=work_directory
    )
    total_energy = calculator.get_potential_energy()

    # Save relaxed geometry
    output_file = Path(work_directory).with_suffix(".xyz")
    write(output_file, calculator)

    elapsed_time = time.time() - start_time
    print(f"     E_total = {total_energy:.6f} eV, Time = {elapsed_time:.2f} s")
    return float(total_energy), elapsed_time


def calculate_adsorption_with_indices(
    optimized_structure: Atoms,
    optimized_molecule: Atoms,
    atom_indices: list,
    work_directory: str,
    height: float = None,
    orientation: list = None,
    calculator: str = "mace",
    yaml_path: str = YAML_PATH,
    ) -> Tuple[float, float]:
    """
    Calculate adsorption energy at site defined by atom indices.
    
    Args:
        optimized_structure: Optimized structure (slab, cluster, etc.)
        optimized_molecule: Optimized gas molecule structure
        atom_indices: List of atom indices defining adsorption site (1-4 atoms)
        work_directory: Directory for calculation files
        height: Adsorption height in Å (None uses default 2.0 Å)
        orientation: Molecule orientation vector (None for automatic)
        calculator: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Tuple of total energy (eV) and calculation time (s)
    """
    os.makedirs(work_directory, exist_ok=True)
    start_time = time.time()

    # Prepare structure
    structure = optimized_structure.copy()
    structure = set_initial_magmoms(structure, kind="cluster_gas")
    structure.set_pbc(True)

    # Prepare adsorbate molecule
    adsorbate = optimized_molecule.copy()
    adsorbate = set_initial_magmoms(
        adsorbate,
        kind="gas",
        formula=adsorbate.get_chemical_formula()
    )
    adsorbate.center()
    adsorbate.set_pbc(False)
    adsorbate.set_cell(None)

    # Place adsorbate using place_adsorbate function
    structure_with_adsorbate = place_adsorbate(
        cluster=structure.copy(),
        adsorbate=adsorbate,
        indices=atom_indices,
        height=height,
        orientation=orientation
    )

    # Set up calculator and calculate energy
    calculator = my_calculator(
        structure_with_adsorbate,
        kind="cluster_gas",
        calculator=calculator,
        yaml_path=yaml_path,
        calc_directory=work_directory
    )
    total_energy = calculator.get_potential_energy()

    # Save results with descriptive filename
    indices_string = "_".join(map(str, atom_indices))
    output_file = Path(work_directory).parent / f"idx_{indices_string}.xyz"
    write(output_file, calculator)

    elapsed_time = time.time() - start_time
    print(f"     Site: {atom_indices}, E_total = {total_energy:.6f} eV, Time = {elapsed_time:.2f} s")
    return float(total_energy), elapsed_time


# ----------------------------------------------------------------------
# Main Calculation Function
# ----------------------------------------------------------------------

def calculate_all_molecules(
    optimized_slab: Atoms,
    slab_energy: float,
    calculator: str = "mace",
    yaml_path: str = YAML_PATH
    ) -> Dict[str, Any]:
    """
    Calculate adsorption energies for all molecules in MOLECULES dictionary.
    
    Args:
        optimized_slab: Optimized slab structure
        slab_energy: Energy of the clean slab (eV)
        calculator: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Dictionary containing all calculation results
    """
    # Dictionary to store all results
    all_results = {}

    # Ensure base directory exists
    os.makedirs(BASE_DIR, exist_ok=True)

    # Process each molecule
    for molecule_name in MOLECULES.keys():
        print(f"=== Processing {molecule_name} ===")

        # Create directories for this molecule
        molecule_directory = os.path.join(BASE_DIR, molecule_name)
        os.makedirs(molecule_directory, exist_ok=True)
        
        gas_directory = os.path.join(molecule_directory, f"{molecule_name}_gas")
        os.makedirs(gas_directory, exist_ok=True)
        
        adsorption_directory = os.path.join(molecule_directory, "adsorption")
        os.makedirs(adsorption_directory, exist_ok=True)

        try:
            # 1. Optimize gas molecule structure
            print(f"1) Optimizing {molecule_name} gas molecule...")
            optimized_molecule, gas_energy = optimize_gas_molecule(
                molecule_name, GAS_BOX, gas_directory, calculator, yaml_path
            )

            # 2. Calculate adsorption energies at different sites
            print(f"2) Calculating adsorption energies at different sites...")
            sites_data = {}
            
            for site in SITES:
                total_energy, elapsed_time = calculate_adsorption_on_site(
                    optimized_slab, optimized_molecule, site, 
                    adsorption_directory, calculator, yaml_path
                )
                sites_data[site] = {
                    "E_total": total_energy,
                    "elapsed_time_s": elapsed_time
                }

            # 3. Calculate adsorption energies and find most stable site
            best_site = None
            best_adsorption_energy = float('inf')

            for site, data in sites_data.items():
                # Calculate adsorption energy: E_ads = E_total - (E_slab + E_gas)
                adsorption_energy = data["E_total"] - (slab_energy + gas_energy)
                sites_data[site]["E_ads"] = adsorption_energy

                # Update most stable site (lowest adsorption energy)
                if adsorption_energy < best_adsorption_energy:
                    best_adsorption_energy = adsorption_energy
                    best_site = site

            # 4. Store results
            all_results[molecule_name] = {
                "E_gas": gas_energy,
                "E_slab": slab_energy,
                "E_ads_best": best_adsorption_energy,
                "best_site": best_site,
                "sites_data": sites_data
            }

            print(f"{molecule_name}: Calculation completed - "
                  f"Best site: {best_site}, "
                  f"Best adsorption energy: {best_adsorption_energy:.6f} eV")

        except Exception as e:
            print(f"Error during calculation of {molecule_name}: {str(e)}")
            all_results[molecule_name] = {"error": str(e)}

    # Save all results to JSON file
    summary_file = os.path.join(BASE_DIR, "all_results.json")
    with open(summary_file, 'w') as f:
        # Convert numpy types for JSON serialization
        results_for_json = convert_numpy_types(all_results)
        json.dump(results_for_json, f, indent=2)

    print(f"\nAll calculations completed. Results saved to {summary_file}")

    return all_results


def search_adsorption_site(
    bulk: Atoms,
    base_dir: str = "result/adsorption_site",
    overwrite: bool = False,
    log_level: str = "INFO",
    calculator: str = "mace",
    adsorbates: Dict[str, Atoms] = None,
    offset: Dict[str, List[Tuple[float, float]]] = None,
    yaml_path: str = None,
) -> Dict[str, Any]:
    """
    Calculate adsorption energies of molecules at various adsorption sites (offset positions)
    and identify the most stable adsorption site.

    Args:
        bulk: Bulk crystal structure
        base_dir: Directory to save calculation results
        overwrite: Whether to overwrite existing calculation results
        log_level: Logging level
        calculator: Calculation type ("vasp", "mace", etc.)
        adsorbates: Dictionary of adsorbate molecules {name: Atoms}
        offset: Offset coordinates for adsorption sites {molecule_name: [(x1,y1), (x2,y2), ...]}
        yaml_path: Path to VASP configuration file

    Returns:
        Dictionary containing the most stable adsorption site information
    """
    # 1. Logging setup
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    # Parameter validation
    if adsorbates is None or len(adsorbates) == 0:
        raise ValueError("Please specify at least one adsorbate molecule")

    # Set default offset values if not specified
    if offset is None:
        offset = {
            name: [(0.0, 0.0), (0.5, 0.0)]  # ontop, bridge
            for name in adsorbates.keys()
        }

    # 2. Directory setup
    base_path = Path(base_dir).resolve()
    base_path.mkdir(parents=True, exist_ok=True)

    bulk_dir = base_path / "bulk"
    slab_dir = base_path / "slab"
    bulk_dir.mkdir(parents=True, exist_ok=True)
    slab_dir.mkdir(parents=True, exist_ok=True)

    # 3. Bulk optimization
    logger = logging.getLogger("adsorption_site_search")
    logger.info("Optimizing bulk structure...")
    optimized_bulk, bulk_energy = optimize_bulk_structure(
        bulk, str(bulk_dir), calculator, yaml_path
    )
    write(str(bulk_dir / "optimized_bulk.xyz"), optimized_bulk)

    # 4. Slab optimization
    logger.info("Optimizing slab structure...")
    optimized_slab, slab_energy = optimize_slab_structure(
        optimized_bulk, str(slab_dir), calculator, yaml_path
    )
    write(str(slab_dir / "optimized_slab.xyz"), optimized_slab)

    # 5. Dictionary to store overall results
    all_results: Dict[str, Any] = {
        "bulk_energy": float(bulk_energy),
        "slab_energy": float(slab_energy),
        "molecules": {}
    }

    # Track the most stable site across all molecules
    most_stable_site = None
    most_stable_energy = float('inf')
    most_stable_molecule = None

    # 6. Calculate for each adsorbate
    for molecule_name, molecule in adsorbates.items():
        logger.info(f"=== Starting adsorption calculation for molecule {molecule_name} ===")

        molecule_dir = base_path / molecule_name
        gas_dir = molecule_dir / f"{molecule_name}_gas"
        adsorption_dir = molecule_dir / "adsorption"

        gas_dir.mkdir(parents=True, exist_ok=True)
        adsorption_dir.mkdir(parents=True, exist_ok=True)

        # 6.1. Gas phase molecule optimization
        gas_json = gas_dir / "opt_gas_result.json"
        xyz_gas = gas_dir / "opt_gas.xyz"

        if gas_json.exists() and not overwrite:
            # Load from existing results
            data = json.load(gas_json.open())
            gas_energy = data["E_opt"]
            optimized_molecule = Atoms.read(xyz_gas)
            logger.info(f"Gas phase molecule energy (existing): {gas_energy:.6f} eV")
        else:
            # 新規計算部分を修正
            # カスタム分子用の一時的な名前を作成
            temp_name = f"custom_{molecule_name}"

            global MOLECULES

            # 一時的に分子辞書に追加
            old_molecules = MOLECULES.copy()
            MOLECULES[temp_name] = molecule.copy()

            try:
                # 正しい引数名で関数を呼び出す
                optimized_molecule, gas_energy = optimize_gas_molecule(
                    molecule_name=temp_name,  # 正しい引数名
                    gas_box_size=GAS_BOX,
                    work_directory=str(gas_dir),
                    calculator=calculator,
                    yaml_path=yaml_path
                )
                optimized_molecule.write(xyz_gas)
                json.dump({"E_opt": float(gas_energy)}, gas_json.open("w"))
                logger.info(f"Gas phase molecule energy (calculated): {gas_energy:.6f} eV")
            finally:
                # 一時的な分子を削除して辞書を元に戻す
                MOLECULES = old_molecules

            optimized_molecule.write(xyz_gas)
            json.dump({"E_opt": float(gas_energy)}, gas_json.open("w"))
            logger.info(f"Gas phase molecule energy (calculated): {gas_energy:.6f} eV")

        # Initialize molecule results dictionary
        molecule_results = {
            "E_gas": float(gas_energy),
            "E_slab": float(slab_energy),
            "offsets": {}
        }

        # 6.2. Calculate at each adsorption site
        offsets_list = offset.get(molecule_name, [(0.0, 0.0), (0.5, 0.0)])
        logger.info(f"Number of adsorption sites to calculate: {len(offsets_list)}")

        for idx, offset_coord in enumerate(offsets_list):
            key = f"ofst_{float(offset_coord[0]):.2f}_{float(offset_coord[1]):.2f}"
            offset_json = adsorption_dir / f"{key}.json"
            work_dir = adsorption_dir / key

            logger.info(f"Adsorption site {idx+1}/{len(offsets_list)}: {key}")

            if offset_json.exists() and (work_dir / ".done").exists() and not overwrite:
                # Load from existing results
                data = json.load(offset_json.open())
                total_energy = data["E_total"]
                elapsed_time = data["elapsed"]
                logger.info(f"  -> Loading existing calculation results")
            else:
                # New calculation
                logger.info(f"  -> Running adsorption calculation...")
                total_energy, elapsed_time = calculate_adsorption_with_offset(
                    optimized_slab, optimized_molecule, offset_coord, str(work_dir),
                    calculator, yaml_path
                )
                json.dump({
                    "E_total": total_energy,
                    "elapsed": elapsed_time
                }, offset_json.open("w"))
                (work_dir / ".done").touch()
                logger.info(f"  -> Calculation completed: {elapsed_time:.1f} seconds")

            # Calculate adsorption energy
            adsorption_energy = total_energy - (slab_energy + gas_energy)

            # Record results
            molecule_results["offsets"][key] = {
                "E_total": float(total_energy),
                "E_ads": float(adsorption_energy),
                "elapsed": float(elapsed_time),
                "coordinates": list(offset_coord)
            }

            logger.info(f"  -> Adsorption energy: {adsorption_energy:.6f} eV")

            # Update most stable site
            if adsorption_energy < most_stable_energy:
                most_stable_energy = adsorption_energy
                most_stable_site = key
                most_stable_molecule = molecule_name

        # Identify the most stable site within this molecule
        if molecule_results["offsets"]:
            best_key, best_data = min(
                molecule_results["offsets"].items(),
                key=lambda x: x[1]["E_ads"]
            )

            molecule_results["best_offset"] = best_key
            molecule_results["best_coordinates"] = best_data["coordinates"]
            molecule_results["E_ads_best"] = float(best_data["E_ads"])
            molecule_results["E_total_best"] = float(best_data["E_total"])

            logger.info(f"Most stable site for molecule {molecule_name}: {best_key}")
            logger.info(f"Most stable adsorption energy: {best_data['E_ads']:.6f} eV")

        # Add molecule results to overall dictionary
        all_results["molecules"][molecule_name] = molecule_results

    # 7. Add overall most stable site information
    if most_stable_site and most_stable_molecule:
        all_results["most_stable_adsorption_site"] = most_stable_site
        all_results["most_stable_adsorption_molecule"] = most_stable_molecule
        all_results["most_stable_adsorption_energy"] = float(most_stable_energy)
        all_results["most_stable_coordinates"] = all_results["molecules"][most_stable_molecule]["offsets"][most_stable_site]["coordinates"]

    # 8. Save results in JSON format
    results_json_path = base_path / "adsorption_site_results.json"
    with open(results_json_path, "w") as f:
        json.dump(convert_numpy_types(all_results), f, indent=2)

    logger.info(f"Calculation results saved: {results_json_path}")

    # 9. Create a concise summary
    summary = {
        "molecules": {},
        "most_stable_adsorption_site": all_results.get("most_stable_adsorption_site", "N/A"),
        "most_stable_adsorption_molecule": all_results.get("most_stable_adsorption_molecule", "N/A"),
        "most_stable_adsorption_energy": all_results.get("most_stable_adsorption_energy", float('inf'))
    }

    for mol_name, mol_data in all_results["molecules"].items():
        if "best_offset" in mol_data:
            summary["molecules"][mol_name] = {
                "best_site": mol_data["best_offset"],
                "best_coordinates": mol_data["best_coordinates"],
                "best_adsorption_energy": mol_data["E_ads_best"]
            }

    summary_path = base_path / "adsorption_site_summary.json"
    with open(summary_path, "w") as f:
        json.dump(convert_numpy_types(summary), f, indent=2)

    logger.info(f"Summary saved: {summary_path}")

    return summary


def attach_modifier_to_surface(
    optimized_slab: Atoms,
    slab_energy: float,
    modifier_name: str,
    modifier_molecule: Atoms,
    offset: Tuple[float, float],
    base_directory: Path,
    calculator: str = "mace",
    yaml_path: str = None,
    overwrite: bool = False,
    ) -> Tuple[Atoms, float]:
    """
    Optimize modifier molecule and adsorb it onto the slab surface.

    Args:
        optimized_slab: Optimized slab structure
        slab_energy: Energy of the slab
        modifier_name: Name of the modifier molecule
        modifier_molecule: Structure of the modifier molecule
        offset: Adsorption position (fractional coordinates)
        base_directory: Directory to save calculation results
        calculator: Calculation type
        yaml_path: Path to VASP configuration file
        overwrite: Whether to overwrite existing calculations

    Returns:
        Slab structure with adsorbed modifier molecule and its energy
    """
    # Directory setup
    modifier_dir = base_directory / "surface_modifier"
    gas_dir = modifier_dir / f"{modifier_name}_gas"
    adsorption_dir = modifier_dir / "adsorption"

    # Create directories
    modifier_dir.mkdir(parents=True, exist_ok=True)
    gas_dir.mkdir(parents=True, exist_ok=True)
    adsorption_dir.mkdir(parents=True, exist_ok=True)

    # 1. Optimize gas phase modifier molecule
    logger = logging.getLogger("adsorption_site_search")
    logger.info(f"Optimizing gas phase {modifier_name}...")

    # Check for existing gas calculation
    gas_json = gas_dir / "opt_result.json"
    xyz_gas = gas_dir / "opt.xyz"

    if gas_json.exists() and xyz_gas.exists() and not overwrite:
        # Load existing results
        with open(gas_json, "r") as f:
            data = json.load(f)
        gas_energy = data["E_opt"]
        optimized_molecule = Atoms()
        optimized_molecule = optimized_molecule.read(str(xyz_gas))
        logger.info(f"Gas phase {modifier_name} energy (existing): {gas_energy:.6f} eV")
    else:
        # New calculation
        # Create temporary name for modifier molecule
        temp_name = f"modifier_{modifier_name}"

        # Declare global variable beforehand
        global MOLECULES

        # Temporarily add to molecule dictionary
        old_molecules = MOLECULES.copy()
        MOLECULES[temp_name] = modifier_molecule.copy()

        try:
            # Optimize gas phase molecule
            optimized_molecule, gas_energy = optimize_gas_molecule(
                molecule_name=temp_name,
                gas_box_size=GAS_BOX,
                work_directory=str(gas_dir),
                calculator=calculator,
                yaml_path=yaml_path
            )
            # Save results
            optimized_molecule.write(str(xyz_gas))
            with open(gas_json, "w") as f:
                json.dump({"E_opt": float(gas_energy)}, f)

            logger.info(f"Gas phase {modifier_name} energy: {gas_energy:.6f} eV")
        finally:
            # Restore global dictionary
            MOLECULES = old_molecules

    # 2. Adsorb modifier molecule onto slab
    logger.info(f"Calculating adsorption of modifier molecule {modifier_name} at position {offset}...")

    key = f"ofst_{offset[0]}_{offset[1]}"
    work_dir = adsorption_dir / key
    adsorption_json = adsorption_dir / f"{key}.json"
    adsorption_xyz = adsorption_dir / f"{key}.xyz"

    if adsorption_json.exists() and adsorption_xyz.exists() and not overwrite:
        # Load existing results
        with open(adsorption_json, "r") as f:
            data = json.load(f)
        total_energy = data["E_total"]
        slab_with_adsorbate = Atoms()
        slab_with_adsorbate = slab_with_adsorbate.read(str(adsorption_xyz))
        logger.info(f"Adsorption calculation (existing): {total_energy:.6f} eV")
    else:
        # New calculation
        work_dir.mkdir(parents=True, exist_ok=True)

        # Create copy of slab
        slab = optimized_slab.copy()
        slab = set_initial_magmoms(slab, kind="slab")

        # Prepare molecule
        adsorbate = optimized_molecule.copy()
        adsorbate = set_initial_magmoms(adsorbate, kind="gas",
                                       formula=adsorbate.get_chemical_formula())
        adsorbate.center()

        # Create slab+adsorbate system
        slab_with_adsorbate = slab.copy()
        slab_with_adsorbate = fix_lower_surface(slab_with_adsorbate)

        # Add molecule
        add_adsorbate(slab_with_adsorbate, adsorbate,
                      ADSORBATE_HEIGHT, position="ontop", offset=offset)

        # 3. Structure optimization and calculation
        calculator = my_calculator(
            atoms=slab_with_adsorbate,
            kind="slab",
            calculator=calculator,
            yaml_path=yaml_path,
            calc_directory=str(work_dir)
        )

        # Energy calculation
        start_time = time.time()
        total_energy = calculator.get_potential_energy()
        elapsed_time = time.time() - start_time

        # Save results
        # Save optimized structure in XYZ format
        write(str(adsorption_xyz), calculator)

        # Save energy information in JSON format
        adsorption_energy = total_energy - (slab_energy + gas_energy)
        with open(adsorption_json, "w") as f:
            json.dump({
                "E_total": float(total_energy),
                "E_gas": float(gas_energy),
                "E_slab": float(slab_energy),
                "E_ads": float(adsorption_energy),
                "elapsed": float(elapsed_time)
            }, f)

        logger.info(f"{modifier_name} adsorption energy: {adsorption_energy:.6f} eV")
        logger.info(f"Calculation time: {elapsed_time:.2f} seconds")

    # Return optimized structure and energy
    return slab_with_adsorbate, total_energy
