import uuid
import sys
import random
import os
import glob
import argparse
import logging
import csv
import yaml
import subprocess
from ase import Atoms
from ase.io import read
from kinetics.microkinetics.orr_and_oer import get_overpotential_for_atoms
from kinetics.utils import make_surface_from_cif, make_barplot, remove_layers
from ase.visualize import view

# Load electron configuration data from YAML file
with open("../../src/kinetics/data/electron_numbers.yaml", "r") as f:
    electron_data = yaml.safe_load(f)

    s_electron_dict = electron_data["s_electrons"]
    p_electron_dict = electron_data["p_electrons"]
    d_electron_dict = electron_data["d_electrons"]
    f_electron_dict = electron_data["f_electrons"]


def get_min_metal_oxygen_distance(surface: Atoms) -> float:
    metal_indices  = [i for i, atom in enumerate(surface) if atom.symbol != "O"]
    oxygen_indices = [i for i, atom in enumerate(surface) if atom.symbol == "O"]

    min_distance = float("inf")
    for metal_idx in metal_indices:
        for oxygen_idx in oxygen_indices:
            distance = surface.get_distance(metal_idx, oxygen_idx)
            if distance < min_distance:
                min_distance = distance

    return min_distance


def write_to_csv(csv_file, row):
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)
    return None


def get_spdf_electrons(surface: Atoms) -> tuple[int]:
    elements = set(surface.get_chemical_symbols())
    s_electrons, p_electrons, d_electrons, f_electrons = 0, 0, 0, 0
    for element in elements:
        if element == "O":
            continue

        if element in s_electron_dict:
            s_electrons += s_electron_dict[element]
        if element in p_electron_dict:
            p_electrons += p_electron_dict[element]
        if element in d_electron_dict:
            d_electrons += d_electron_dict[element]
        if element in f_electron_dict:
            f_electrons += f_electron_dict[element]

        if s_electrons + p_electrons + d_electrons + f_electrons == 0:
            raise ValueError(f"{element} not in the electron number list")

    return s_electrons, p_electrons, d_electrons, f_electrons


def clean():
    # cleanup past calculation
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_script = os.path.join(script_dir, "clean.sh")
    try:
        subprocess.run(["bash", clean_script])
    except Exception as e:
        logger.error(f"Failed to executre {clean_script}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.info("Start calculation")

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_sample", default=10, help="number of samples")
    parser.add_argument("--calculator", default="mace", help="energy calculator")
    parser.add_argument("--repeat", default="222", help="repeat unit cell in xyz")
    parser.add_argument("--vacuum", default=7.0, help="vacuum layer in Angstrom")
    args = parser.parse_args()
    max_sample = int(args.max_sample)
    calculator = args.calculator
    repeat = [int(char) for char in args.repeat]
    vacuum = float(args.vacuum)

    # cleanup past calculation
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_script = os.path.join(script_dir, "clean.sh")
    try:
        subprocess.run(["bash", clean_script])
    except Exception as e:
        logger.error(f"Failed to executre {clean_script}")

    # when writing to csv file
    csv_file = "output.csv"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["formula", "cell_volume",
                         "s_electrons", "p_electrons", "d_electrons", "f_electrons",
                         "min_M_O_distance", "overpotential_in_eV"])

    # set random seed for reproducibility
    random.seed(0)

    materials = []
    etas = []

    reaction_file = "oer.txt"
    # reaction_file = "oer2.txt"
    # energy_shift = [-0.32, -0.54, -0.47, -0.75]
    energy_shift = [0, 0, 0, -4.92]

    directory = "/ABO3_single/"
    # directory = "/ABO3_cif_large/"
    # directory = "/ABO3_cif/"
    # directory = "/ABO3_Mn_cif/"

    cif_files = sorted(glob.glob(os.getcwd() + directory + "*.cif"))
    cif_files = random.sample(cif_files, min(len(cif_files), max_sample))
    possible_indices = [[0, 0, 1]]
    # possible_indices = [[0, 0, 1], [1, 1, 0], [1, 1, 1]]

    for cif_file in cif_files[:max_sample]:
        for miller_indices in possible_indices:
            clean()

            if not os.path.isfile(cif_file):
                msg = f"Could not find file {cif_file}"
                logger.error(msg)
                raise ValueError(msg)

            bulk = read(cif_file)
            material = os.path.basename(cif_file).split("_")[1].split(".")[0]
            surface = make_surface_from_cif(cif_file, indices=[0, 0, 1], repeat=repeat, vacuum=vacuum)
            eta = get_overpotential_for_atoms(surface=surface, calculator=calculator,
                                              input_yaml="vasp.yaml",
                                              reaction_type="oer", reaction_file=reaction_file)

            # getting descriptors
            formula = surface.get_chemical_formula()
            cell_volume = bulk.get_volume()  # volume of the bulk unit cell
            s_electrons, p_electrons, d_electrons, f_electrons = get_spdf_electrons(surface)
            min_M_O_distance = get_min_metal_oxygen_distance(surface)

            if eta is not None:
                logger.info(f"material: {material:12.10s} eta: {eta:5.3f} eV")
                materials.append(material)
                etas.append(eta)
                write_to_csv(csv_file, [formula, cell_volume, s_electrons, p_electrons,
                                        d_electrons, f_electrons, min_M_O_distance, eta])
            else:
                logger.error(f"failed for {material}")

        # make_barplot(labels=materials, values=etas, threshold=1000)

    logger.info("All done")
