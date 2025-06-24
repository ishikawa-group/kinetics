import os
import json
import logging
import argparse
from datetime import datetime
from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar, Kpoints


def setup_test_logging(log_dir="output_logs"):
    """
    Set up test logging

    Args:
        log_dir (str): Directory for log files

    Returns:
        logger (Logger): Logger object    
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mp_test_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command line arguments

    Args:
        None

    Returns:
        args (Namespace): Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate VASP input files for ABO3 materials from Materials Project"
    )

    # Required arguments
    parser.add_argument(
        "--api-key",
        required=True,
        # default="YOUR_API_KEY",
        help="Materials Project API key"
    )
    parser.add_argument(
        "--potcar-dir",
        required=True,
        default="/path/to/POTENTIALS",
        help="Directory containing POTCAR files"
    )

    # Optional arguments
    parser.add_argument(
        "--max-materials",
        type=int,
        # required=True,
        default=5,  # set 5 just for testing
        help="Maximum number of materials to process (default: 5)"
    )
    parser.add_argument(
        "--crystal-system",
        # required=True,
        default="cubic",
        choices=["cubic", "tetragonal", "orthorhombic", "hexagonal", "trigonal", "monoclinic", "triclinic", None],
        help="Crystal system to filter materials (default: cubic)"
    )
    parser.add_argument(
        "--vasp-path",
        # required=True,
        default="/gs/fs/tga-ishikawalab/vasp/vasp.6.4.3/bin/vasp_std",
        help="Path to VASP executable"
    )
    parser.add_argument(
        "--base-dir",
        default="vasp_calculations",
        help="Base directory for output files (default: vasp_calculations)"
    )
    parser.add_argument(
        "--preferred-potcar",
        type=json.loads,
        help="JSON string of preferred POTCAR types (e.g., '{\"Fe\":\"Fe_pv\",\"O\":\"O\"}')"
    )
    parser.add_argument(
        "--log-dir",
        default="output_logs",
        help="Directory for log files (default: output_logs)"
    )

    return parser.parse_args()


def get_available_potcar_types(element, potcar_dir):
    """
    Get available POTCAR types for a given element

    Args:
        element (str): Element symbol
        potcar_dir (str): Directory containing POTCAR files

    Returns:
        element_folders (list): List of available POTCAR types
    """
    element_folders = []

    # Scan all folders in the directory
    for folder in os.listdir(potcar_dir):
        # Remove all suffixes (_GW, _sv_GW, etc.) to get the base element name
        base_element = folder.split('_')[0]

        # Check if it is the element we are looking for
        if base_element == element:
            potcar_path = os.path.join(potcar_dir, folder, 'POTCAR')
            if os.path.exists(potcar_path):
                element_folders.append(folder)

    return sorted(element_folders)  # Sort to ensure consistency


def select_potcar_type(element, available_types, preferred_types=None):
    """
    Select POTCAR type based on priority

    Args:
        element (str): Element symbol
        available_types (list): List of available POTCAR types
        preferred_types (dict): Dictionary of preferred POTCAR types

    Returns:
        selected_type (str): Selected POTCAR type
    """
    if not available_types:
        return None

    if preferred_types and element in preferred_types:
        if preferred_types[element] in available_types:
            return preferred_types[element]

    # Priority order
    # Reference: https://www.vasp.at/wiki/index.php/Available_pseudopotentials?utm_source=chatgpt.com#potpaw.64_(latest,_recommended)
    priority_suffixes = [
        '_pv',         # With p valence electrons
        '_sv',         # With semi-core valence electrons
        '',           # Standard version
        '_GW',        # GW version
        '_sv_GW',     # GW with semi-core valence electrons
        '_h',         # Hard version
        '_s'          # Soft version
    ]

    # Search by priority
    for suffix in priority_suffixes:
        for pot_type in available_types:
            if pot_type == f"{element}{suffix}":
                return pot_type

    return available_types[0]


def get_ordered_elements(structure):
    """
    Get ordered list of elements in the structure

    Args:
        structure (Structure): Input structure

    Returns:
        elements (list): List of elements in the structure
    """
    elements = []
    for site in structure:
        element = str(site.specie)
        if element not in elements:
            elements.append(element)
    return elements


def combine_potcar_files(element_list, potcar_dir, logger, preferred_types=None):
    """
    Combine POTCAR files

    Args:
        element_list (list): List of elements
        potcar_dir (str): Directory containing POTCAR files
        logger (Logger): Logger object
        preferred_types (dict): Dictionary of preferred POTCAR types

    Returns:
        combined_content (str): Combined content of POTCAR files
        selected_types (dict): Selected POTCAR types for each element
    """
    logger.info(f"Combining POTCAR files for elements: {element_list}")

    # Record available pseudopotential types
    available_types = {}
    for element in element_list:
        types = get_available_potcar_types(element, potcar_dir)
        available_types[element] = types
        logger.info(f"Available POTCAR types for {element}: {types}")

    combined_content = []
    selected_types = {}
    missing_elements = []

    for element in element_list:
        if not available_types[element]:
            logger.error(f"No POTCAR found for element {element}")
            missing_elements.append(element)
            continue

        selected_type = select_potcar_type(element, available_types[element], preferred_types)
        selected_types[element] = selected_type

        potcar_path = os.path.join(potcar_dir, selected_type, 'POTCAR')
        logger.info(f"Using POTCAR type '{selected_type}' for element {element}")

        try:
            with open(potcar_path, 'r') as f:
                content = f.read()
                combined_content.append(content)
        except Exception as e:
            logger.error(f"Error reading POTCAR for {selected_type}: {str(e)}")
            missing_elements.append(element)

    if missing_elements:
        raise FileNotFoundError(f"Missing POTCAR files for elements: {', '.join(missing_elements)}")

    return ''.join(combined_content), selected_types


def generate_incar(material_id, formula):
    """
    Generate the content of the INCAR file

    Args:
        material_id (str): Material ID
        formula (str): Chemical formula

    Returns:
        incar_content (str): Content of the INCAR file
    """
    incar_dict = {
        "SYSTEM": f"{formula}_{material_id}",
        "PREC": "Normal",
        "ISTART": 0,
        "ICHARG": 2,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "ENCUT": 500,
        "EDIFF": 1E-5,
        "GGA": "PE",
        "LASPH": ".TRUE.",
        "NSW": 50,
        "IBRION": 2,
        "ISIF": 3,
        "EDIFFG": -0.02,
        "LWAVE": ".FALSE.",
        "LCHARG": ".FALSE."
    }

    incar_content = []
    for key, value in incar_dict.items():
        incar_content.append(f"{key} = {value}")

    return "\n".join(incar_content)


def generate_run_script(material_id, vasp_path):
    """
    Generate the job submission script

    Args:
        material_id (str): Material ID
        vasp_path (str): Path to VASP executable

    Returns:
        script_content (str): Content of the job submission script
    """
    script_content = f"""#!/bin/sh
#$ -cwd
# Uses one node of O-type
#$ -l cpu_40=1
#$ -l h_rt=10:00:00
#$ -N {material_id}
# pass to the VASP executable file
PRG={vasp_path}

. /etc/profile.d/modules.sh
module load cuda
module load intel
# Loading Intel MPI
module load intel-mpi

# Uses 8 processes with MPI
mpiexec.hydra -ppn 8 -n 8 ${{PRG}} >& vasp.out"""

    return script_content


def prepare_vasp_inputs(structure, material_id, formula, potcar_dir, vasp_path, base_dir,
                        preferred_potcar_types=None, logger=None):
    """
    Prepare all VASP input files

    Args:
        structure (Structure): Input structure
        material_id (str): Material ID
        formula (str): Chemical formula
        potcar_dir (str): Directory containing POTCAR files
        vasp_path (str): Path to VASP executable
        base_dir (str): Base directory for output files
        preferred_potcar_types (dict): Dictionary of preferred POTCAR types
        logger (Logger): Logger object

    Returns:
        vasp_inputs (dict): Dictionary of VASP input file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Create material directory
    material_dir = os.path.join(base_dir, material_id)
    os.makedirs(material_dir, exist_ok=True)

    try:
        # Save POSCAR
        poscar = Poscar(structure)
        poscar_path = os.path.join(material_dir, "POSCAR")
        poscar.write_file(poscar_path)
        logger.info(f"Saved POSCAR to {poscar_path}")

        # Generate and save POTCAR
        ordered_elements = get_ordered_elements(structure)
        potcar_content, selected_types = combine_potcar_files(
            ordered_elements,
            potcar_dir,
            logger,
            preferred_potcar_types
        )

        potcar_path = os.path.join(material_dir, "POTCAR")
        with open(potcar_path, 'w') as f:
            f.write(potcar_content)
        logger.info(f"Saved POTCAR to {potcar_path}")

        # Generate and save INCAR
        incar_content = generate_incar(material_id, formula)
        incar_path = os.path.join(material_dir, "INCAR")
        with open(incar_path, 'w') as f:
            f.write(incar_content)
        logger.info(f"Saved INCAR to {incar_path}")

        # Generate and save KPOINTS
        kpoints = Kpoints.gamma_automatic((6, 6, 6))
        kpoints_path = os.path.join(material_dir, "KPOINTS")
        kpoints.write_file(kpoints_path)
        logger.info(f"Saved KPOINTS to {kpoints_path}")

        # Generate and save run.sh
        run_script_content = generate_run_script(material_id, vasp_path)
        run_script_path = os.path.join(material_dir, "run.sh")
        with open(run_script_path, 'w') as f:
            f.write(run_script_content)
        # Set execute permissions
        os.chmod(run_script_path, 0o755)
        logger.info(f"Saved run.sh to {run_script_path}")

        return {
            "poscar_path": poscar_path,
            "potcar_path": potcar_path,
            "incar_path": incar_path,
            "kpoints_path": kpoints_path,
            "run_script_path": run_script_path,
            "potcar_types": selected_types
        }

    except Exception as e:
        logger.error(f"Error preparing VASP inputs for {material_id}: {str(e)}")
        raise


def process_abo3_materials(api_key, vasp_path, potcar_dir, max_materials, crystal_system="cubic",
                           preferred_potcar_types=None):
    """
    Main function to process ABO3 materials

    Args:
        api_key (str): Materials Project API key
        potcar_dir (str): Directory containing POTCAR files
        max_materials (int): Maximum number of materials to process
        crystal_system (str): Crystal system to filter materials
        preferred_potcar_types (dict): Dictionary of preferred POTCAR types

    Returns:
        processed_materials (list): List of processed materials
    """
    logger = setup_test_logging()
    logger.info("Starting ABO3 materials processing")
    logger.info(f"Parameters: max_materials={max_materials}, crystal_system={crystal_system}")
    logger.info(f"POTCAR directory: {potcar_dir}")

    try:
        with MPRester(api_key) as mpr:
            # Use materials.summary path and correct search parameters
            docs = mpr.materials.summary.search(
                num_sites=5,
                elements=["O"],
                fields=["material_id", "formula_pretty", "structure"]
            )

            processed_materials = []
            count = 0

            for doc in docs:
                structure = doc.structure
                composition = structure.composition

                # Check if it is ABO3
                if composition["O"] == 3 and len(composition) == 3:
                    analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
                    crystal_sys = analyzer.get_crystal_system().lower()

                    if crystal_system is None or crystal_sys == crystal_system.lower():
                        try:
                            # Prepare VASP input files
                            vasp_inputs = prepare_vasp_inputs(
                                structure=structure,
                                material_id=doc.material_id,
                                formula=doc.formula_pretty,
                                potcar_dir=potcar_dir,
                                base_dir="vasp_calculations",
                                preferred_potcar_types=preferred_potcar_types,
                                logger=logger,
                                vasp_path=vasp_path
                            )

                            material_info = {
                                "material_id": doc.material_id,
                                "formula": doc.formula_pretty,
                                "crystal_system": crystal_sys,
                                "files": vasp_inputs
                            }

                            processed_materials.append(material_info)
                            logger.info(f"Successfully processed material: {doc.material_id}")

                            count += 1
                            if count >= max_materials:
                                break

                        except Exception as e:
                            logger.error(f"Error processing material {doc.material_id}: {str(e)}")
                            continue

            # Save processing results
            results_file = "processed_materials.json"
            with open(results_file, 'w') as f:
                json.dump(processed_materials, f, indent=4)
            logger.info(f"Saved processing results to {results_file}")

            return processed_materials

    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        raise


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()

    try:
        # Process materials with command line arguments
        materials = process_abo3_materials(
            api_key=args.api_key,
            potcar_dir=args.potcar_dir,
            max_materials=args.max_materials,
            crystal_system=args.crystal_system,
            preferred_potcar_types=args.preferred_potcar,
            vasp_path=args.vasp_path
        )

        # # Print summary of results
        # print("\nProcessing Summary:")
        # print(f"Total materials processed: {len(materials)}")
        # for material in materials:
        #     print(f"\nMaterial ID: {material['material_id']}")
        #     print(f"Formula: {material['formula']}")
        #     print(f"Crystal System: {material['crystal_system']}")
        #     print(f"Files generated:")
        #     for file_type, file_path in material['files'].items():
        #         if file_type != 'potcar_types':
        #             print(f"- {file_type}: {file_path}")
        #     print(f"Selected POTCAR types: {material['files']['potcar_types']}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
