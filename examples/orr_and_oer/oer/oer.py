import uuid
import sys
import random
import os
import glob
import argparse
import logging
import csv
from ase import Atoms
from ase.io import read
from kinetics.microkinetics.orr_and_oer import get_overpotential_for_atoms
from kinetics.microkinetics.utils import make_surface_from_cif, make_barplot

s_electron_dict = {"Mg": 2, "Ca": 2, "Mn": 2, "Fe": 2, "Sn": 2, "Ba": 2, "Si": 2, "Ge": 2, "Y": 2,
                   "Ir": 2, "Cd": 2, "Ru": 1, "Ti": 2, "Rh": 1, "Os": 2, "Hg": 2, "Pb": 2, "Sb": 2,
                   "Cu": 1, "Ag": 1, "Mo": 1, "Na": 1, "K": 1, "Li": 1, "Tl": 2, "Rb": 1, "Al": 2,
                   "Pd": 0, "Cs": 1, "Tc": 2, "Np": 2, "W": 2, "Be": 2, "Nd": 2, "Sr": 2, "As": 2,
                   "B": 2, "Re": 2, "Au": 1, "La": 2, "Sc": 2, "V": 2, "Pt": 1, "Te": 2, "Nb": 1,
                   "Zr": 3, "Ni": 2, "Zn": 2, "Ta": 2, "Cr": 1, "Tb": 2, "I": 2, "Bi": 2, "Ga": 2,
                   "In": 2, "Co": 2, "Ce": 2, "Gd": 2}

p_electron_dict = {"Mg": 0, "Ca": 0, "Mn": 0, "Fe": 0, "Sn": 2, "Ba": 0, "Si": 2, "Ge": 2, "Y": 0,
                   "Ir": 0, "Cd": 0, "Ru": 0, "Ti": 0, "Rh": 0, "Os": 0, "Hg": 0, "Pb": 2, "Sb": 3,
                   "Cu": 0, "Ag": 0, "Mo": 0, "Na": 0, "K": 0, "Li": 0, "Tl": 0, "Rb": 0, "Al": 1,
                   "Pd": 0, "Cs": 0, "Tc": 0, "Np": 0, "W": 0, "Be": 0, "Nd": 0, "Sr": 0, "As": 3,
                   "B": 1, "Re": 0, "Au": 0, "La": 0, "Sc": 0, "V": 0, "Pt": 0, "Te": 4, "Nb": 0,
                   "Zr": 0, "Ni": 0, "Zn": 0, "Ta": 0, "Cr": 0, "Tb": 0, "I": 5, "Bi": 3, "Ga": 1,
                   "In": 1, "Co": 0, "Ce": 0, "Gd": 0}

d_electron_dict = {"Mg": 0, "Ca": 0, "Mn": 5, "Fe": 6, "Sn": 10, "Ba": 0, "Si": 0, "Ge": 10, "Y": 1,
                   "Ir": 7, "Cd": 10, "Ru": 7, "Ti": 2, "Rh": 8, "Os": 6, "Hg": 10, "Pb": 10, "Sb": 10,
                   "Cu": 10, "Ag": 10, "Mo": 5, "Na": 0, "K": 0, "Li": 0, "Tl": 3, "Rb": 0, "Al": 0,
                   "Pd": 10, "Cs": 0, "Tc": 5, "Np": 1, "W": 4, "Be": 0, "Nd": 0, "Sr": 0, "As": 10,
                   "B": 0, "Re": 5, "Au": 10, "La": 1, "Sc": 1, "V": 3, "Pt": 9, "Te": 10, "Nb": 4,
                   "Zr": 3, "Ni": 8, "Zn": 10, "Ta": 3, "Cr": 5, "Tb": 0, "I": 10, "Bi": 10, "Ga": 10,
                   "In": 10, "Co": 7, "Ce": 1, "Gd": 1}

f_electron_dict = {"Mg": 0, "Ca": 0, "Mn": 0, "Fe": 0, "Sn": 0, "Ba": 0, "Si": 0, "Ge": 0, "Y": 0,
                   "Ir": 14, "Cd": 0, "Ru": 0, "Ti": 0, "Rh": 0, "Os": 14, "Hg": 14, "Pb": 14, "Sb": 0,
                   "Cu": 0, "Ag": 0, "Mo": 0, "Na": 0, "K": 0, "Li": 0, "Tl": 14, "Rb": 0, "Al": 0,
                   "Pd": 0, "Cs": 0, "Tc": 0, "Np": 4, "W": 14, "Be": 0, "Nd": 4, "Sr": 0, "As": 0,
                   "B": 0, "Re": 14, "Au": 14, "La": 0, "Sc": 0, "V": 0, "Pt": 14, "Te": 0, "Nb": 0,
                   "Zr": 0, "Ni": 0, "Zn": 0, "Ta": 14, "Cr": 0, "Tb": 9, "I": 0, "Bi": 14, "Ga": 0,
                   "In": 0, "Co": 0, "Ce": 1, "Gd": 7}


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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.info("Start calculation")

    # when writing to csv file
    csv_file = "output.csv"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["formula", "cell_volume",
                         "s_electrons", "p_electrons", "d_electrons", "f_electrons",
                         "min_M_O_distance", "overpotential_in_eV"])

    # surface model parameters
    repeat = [2, 2, 2]
    vacuum = 6.5
    max_sample = 20

    materials = []
    etas = []

    reaction_file = "oer.txt"
    energy_shift = [-0.32, -0.54, -0.47, -0.75]
    calculator = "m3gnet"  # ( "vasp" | "m3gnet" | "mattersim" )

    # directory = None
    directory = "/ABO3_cif_large/"
    # directory = "/ABO3_cif/"
    # directory = "/ABO3_Mn_cif/"

    if directory is not None:  # Directory mode
        cif_files = sorted(glob.glob(os.getcwd() + directory + "*.cif"))
        cif_files = random.sample(cif_files, min(len(cif_files), max_sample))

        for cif_file in cif_files[:max_sample]:
            if not os.path.isfile(cif_file):
                logger.info(f"Could not found file: {cif_file}")
                raise ValueError

            bulk = read(cif_file)
            material = os.path.basename(cif_file).split("_")[1].split(".")[0]
            surface = make_surface_from_cif(cif_file, indices=[0, 0, 1], repeat=repeat, vacuum=vacuum)
            eta = get_overpotential_for_atoms(surface=surface, calculator=calculator,
                                              reaction_type="oer", reaction_file=reaction_file)

            # descriptors
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
                logger.info(f"failed for {material}")

        make_barplot(labels=materials, values=etas, threshold=1000)

    else:  # Single file mode
        cif_file = "CaMn2O4_ICSD_CollCode280514.cif"
        bulk = read(cif_file)

        if not os.path.isfile(cif_file):
            raise ValueError(f"Could not found file: {cif_file}")

        logger.info(f"Calculating {cif_file}")
        material = os.path.basename(cif_file).split("_")[1].split(".")[0]

        surface = make_surface_from_cif(cif_file, indices=[0, 0, 1], repeat=repeat, vacuum=vacuum)

        # descriptors
        formula = surface.get_chemical_formula()
        cell_volume = bulk.get_volume()  # volume of the bulk unit cell
        s_electrons, p_electrons, d_electrons, f_electrons = get_spdf_electrons(surface)

        eta = get_overpotential_for_atoms(surface=surface, calculator=calculator,
                                          reaction_type="oer", reaction_file=reaction_file)

        if eta is None:
            logger.info(f"failed for {material}")
        else:
            logger.info(f"file = {material:16.14s}, eta = {eta:5.3f} eV")
            write_to_csv(csv_file, [formula, cell_volume, s_electrons, p_electrons,
                                    d_electrons, f_electrons, min_M_O_distance, eta])

    logger.info("All done")
