import uuid
import sys
import os
import glob
import argparse
import logging
from kinetics.microkinetics.utils import (
    make_surface_from_cif, remove_layers, replace_element,
    fix_lower_surface, sort_atoms_by_z, add_data_to_jsonfile, make_barplot
)
from kinetics.microkinetics.get_reaction_energy import get_reaction_energy
from kinetics.microkinetics.orr_and_oer import get_overpotential_oer_orr


def get_overpotential_for_cif(cif_file=None, dirname=None):
    # replace_percent = 0

    repeat = [2, 2, 2]

    vacuum = 6.5
    surface = make_surface_from_cif(cif_file, indices=[0, 0, 1], repeat=repeat, vacuum=vacuum)

    surface, count = sort_atoms_by_z(surface)
    lowest_z = surface[0].position[2]
    surface.translate([0, 0, -lowest_z + 0.1])

    # surface = remove_layers(surface, element="Ca", layers_to_remove=1)
    # surface = remove_layers(surface, element="O", layers_to_remove=1)
    # surface = replace_element(surface, from_element="Mn", to_element="Cr", percent=100)

    surface = remove_layers(surface, element="O", layers_to_remove=1)  # remove O
    surface = remove_layers(surface, layers_to_remove=1)  # remove A cation
    surface = fix_lower_surface(surface)

    # reaction_file = "orr_alkaline.txt"  # not really good on first step
    reaction_file = "orr_alkaline2.txt"; energy_shift = [-0.32, -0.54, -0.47, -0.75]
    # reaction_file = "orr_alkaline2.txt"; energy_shift = [-0.32+0.75, -0.54+0.32, -0.47+0.54, -0.75+0.47]
    # reaction_file = "orr_alkaline3.txt"; energy_shift = [-0.32+0.75-4.92, -0.54+0.32, -0.47+0.54, -0.75+0.47]
    # reaction_file = "orr_alkaline3.txt"; energy_shift = [-0.32-4.92, -0.54, -0.47, -0.75]
    # reaction_file = "orr_alkaline3.txt"; energy_shift = [-4.92, 0, 0, 0]

    calculator = "m3gnet"

    deltaEs = get_reaction_energy(reaction_file=reaction_file, surface=surface, calculator=calculator,
                                  input_yaml="tmp.yaml", verbose=True, dirname=dirname)
    eta = get_overpotential_oer_orr(reaction_file=reaction_file, deltaEs=deltaEs,
                                    reaction_type="orr", verbose=True, energy_shift=energy_shift)

    # Save results
    data = {
        "unique_id": str(uuid.uuid4()),
        "chemical_formula": surface.get_chemical_formula(),
        "atomic_numbers": surface.get_atomic_numbers().tolist(),
        "overpotential": eta,
        "status": "done"
    }
    add_data_to_jsonfile(data=data, jsonfile="surf.json")

    return eta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", type=str, default="0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.info("Start calculation")

    materials = []
    etas = []

    # directory = None
    # directory = "/ABO3_cif_large/"
    # directory = "/ABO3_cif/"
    directory = "/ABO3_Mn_cif/"

    if directory is not None:  # Directory mode
        cif_files = sorted(glob.glob(os.getcwd() + directory + "*.cif"))

        end = args.end if args.end is not None else len(cif_files)
        logger.info(f"Found {len(cif_files)} files, and do calculation from {args.start} to {end}.")

        for cif_file in cif_files[args.start:end]:
            if not os.path.isfile(cif_file):
                logger.info(f"Could not found file: {cif_file}")
                continue

            eta = get_overpotential_for_cif(cif_file=cif_file, dirname=args.dirname)
            material = os.path.basename(cif_file).split("_")[1].split(".")[0]

            if eta is not None:
                logger.info(f"material: {material:12.10s} eta: {eta:5.3f} eV")
                materials.append(material)
                etas.append(eta)
            else:
                logger.info(f"failed for {material}")

    else:  # Single file mode
        cif_file = "CaMn2O4_ICSD_CollCode280514.cif"
        if not os.path.isfile(cif_file):
            logger.info(f"Could not found file: {cif_file}")
        else:
            eta = get_overpotential_for_cif(cif_file=cif_file, dirname=args.dirname)
            material = os.path.basename(cif_file).split("_")[1].split(".")[0]

            if eta is None:
                logger.info(f"failed for {material}")
            else:
                logger.info(f"file = {material:16.14s}, eta = {eta:5.3f} eV")
                materials.append(material)
                etas.append(eta)

    make_barplot(labels=materials, values=etas, threshold=1000)
    logger.info("All done")
