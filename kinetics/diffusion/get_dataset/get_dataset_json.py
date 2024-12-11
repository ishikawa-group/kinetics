import os
import sys
import json
import numpy as np
from pymatgen.io.vasp import Poscar, Outcar, Vasprun
import logging
from datetime import datetime
from pathlib import Path

def setup_logging():
    """
    Set up logging configuration
    
    Args:
        None
    
    Returns:
        logger: Logger object
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_vasp_files(poscar_path, outcar_path, vasprun_path):
    """Parse VASP output files and return the data"""
    # Check if files exist
    for file_path in [poscar_path, outcar_path, vasprun_path]:
        if not os.path.isfile(file_path):
            print(f"Error: File does not exist - {file_path}")
            return None

    # Read POSCAR or CONTCAR file
    try:
        poscar = Poscar.from_file(poscar_path)
        structure = poscar.structure
        structure_data = {
            "lattice": structure.lattice.matrix.tolist(),
            "species": [str(sp) for sp in structure.species],
            "coords": structure.frac_coords.tolist()
        }
    except Exception as e:
        print(f"Error: Unable to read POSCAR file - {e}")
        return None

    # Initialize data for this structure
    forces = None
    energy = None
    stress = None

    # Read OUTCAR file to extract force information
    try:
        outcar = Outcar(outcar_path)
        if hasattr(outcar, 'ionic_steps') and outcar.ionic_steps:
            last_step_outcar = outcar.ionic_steps[-1]
            if 'forces' in last_step_outcar and last_step_outcar['forces'] is not None:
                forces = last_step_outcar['forces']
                if isinstance(forces, np.ndarray):
                    forces = forces.tolist()
    except Exception as e:
        print(f"Error: Unable to read OUTCAR file - {e}")
        return None

    # Read vasprun.xml file
    try:
        vasprun = Vasprun(vasprun_path, parse_potcar_file=False, parse_dos=False, parse_eigen=False)
        energy = vasprun.final_energy

        ionic_steps = vasprun.ionic_steps
        if ionic_steps:
            last_step_vasprun = ionic_steps[-1]

            # Extract stress tensor
            if "stress" in last_step_vasprun:
                stress_tensor = last_step_vasprun["stress"]
                if isinstance(stress_tensor, list):
                    stress_array = np.array(stress_tensor, dtype=float)
                    stress = (stress_array * 0.1).flatten().tolist()  # Convert kBar to GPa

            # Get forces from vasprun if not already extracted
            if forces is None and "forces" in last_step_vasprun:
                forces = last_step_vasprun["forces"]
                if isinstance(forces, np.ndarray):
                    forces = forces.tolist()

    except Exception as e:
        print(f"Error: Unable to read vasprun.xml file - {e}")
        return None

    if energy is None or forces is None or stress is None:
        return None

    return {
        "structure": structure_data,
        "energy": energy,
        "forces": forces,
        "stress": stress
    }

def process_all_calculations(base_dir, output_file):
    """Process all VASP calculations and combine into one dataset"""
    logger = setup_logging()
    logger.info("Preparing datasets...")
    
    # Initialize the combined dataset structure
    dataset = {
        "structures": [],
        "labels": {
            "energies": [],
            "stresses": [],
            "forces": []
        }
    }

    success_count = 0
    failed_count = 0

    # Process all directories containing VASP outputs
    for outcar_path in sorted(Path(base_dir).glob("**/OUTCAR")):
        calc_dir = outcar_path.parent
        logger.info(f"Processing directory: {calc_dir}")
        
        # Set up file paths
        poscar_path = os.path.join(calc_dir, "CONTCAR")
        if not os.path.exists(poscar_path):
            poscar_path = os.path.join(calc_dir, "POSCAR")
        outcar_path = str(outcar_path)
        vasprun_path = os.path.join(calc_dir, "vasprun.xml")
        
        # Parse the files
        result = parse_vasp_files(poscar_path, outcar_path, vasprun_path)
        
        if result is not None:
            # Add to the combined dataset
            dataset["structures"].append(result["structure"])
            dataset["labels"]["energies"].append(result["energy"])
            dataset["labels"]["stresses"].append(result["stress"])
            dataset["labels"]["forces"].append(result["forces"])
            success_count += 1
        else:
            failed_count += 1
            logger.warning(f"Failed to process: {calc_dir}")

    # Save the combined dataset
    try:
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        logger.info(f"Combined dataset saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save dataset: {str(e)}")
        return False

    # Print summary
    logger.info("\nProcessing summary:")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Processing failed: {failed_count}")
    logger.info(f"Total attempts: {success_count + failed_count}")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <vasp_calculations_dir> [output_file]")
        sys.exit(1)

    base_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "dataset.json"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)

    process_all_calculations(base_dir, output_file)

if __name__ == "__main__":
    main()