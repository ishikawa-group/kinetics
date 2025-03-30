def get_overpotential_for_cif(cif_file=None, dirname=None):
    import numpy as np
    import json
    import uuid
    import argparse
    from kinetics.microkinetics.utils import make_surface_from_cif
    from kinetics.microkinetics.utils import remove_layers
    from kinetics.microkinetics.utils import replace_element
    from kinetics.microkinetics.utils import fix_lower_surface
    from kinetics.microkinetics.utils import sort_atoms_by_z
    from kinetics.microkinetics.utils import add_data_to_jsonfile
    from kinetics.microkinetics.utils import make_barplot
    from kinetics.microkinetics.get_reaction_energy import get_reaction_energy
    from kinetics.microkinetics.orr_and_oer import get_overpotential_oer_orr 
    from ase.visualize import view
    import warnings
    warnings.filterwarnings("ignore")

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

    energy_shift = [0]*4

    # reaction_file = "orr_alkaline.txt"  # not really good on first step
    reaction_file = "orr_alkaline2.txt"; energy_shift = [-0.32, -0.54, -0.47, -0.75]
    # reaction_file = "orr_alkaline2.txt"; energy_shift = [-0.32+0.75, -0.54+0.32, -0.47+0.54, -0.75+0.47]
    # reaction_file = "orr_alkaline3.txt"; energy_shift = [-0.32+0.75-4.92, -0.54+0.32, -0.47+0.54, -0.75+0.47]
    # reaction_file = "orr_alkaline3.txt"; energy_shift = [-0.32-4.92, -0.54, -0.47, -0.75]
    # reaction_file = "orr_alkaline3.txt"; energy_shift = [-4.92, 0, 0, 0]

    calculator = "m3gnet"

    deltaEs = get_reaction_energy(reaction_file=reaction_file, surface=surface, calculator=calculator, input_yaml="tmp.yaml", verbose=True, dirname=dirname)

    if deltaEs is None:
        return None

    eta = get_overpotential_oer_orr(reaction_file=reaction_file, deltaEs=deltaEs, reaction_type="orr", verbose=True, energy_shift=energy_shift)
    eta = np.abs(eta)

    jsonfile = "surf.json"

    atomic_numbers = surface.get_atomic_numbers()
    formula = surface.get_chemical_formula()
    unique_id = uuid.uuid4()
    data = {"unique_id": str(unique_id), "chemical_formula": formula, "atomic_numbers": atomic_numbers.tolist(), "overpotential": eta, "status": "done"}

    add_data_to_jsonfile(data=data, jsonfile=jsonfile)

    return eta


if __name__ == "__main__":
    import sys
    import os
    import glob
    from ase.io import read
    import logging
    import argparse
    from kinetics.microkinetics.utils import make_barplot

    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", type=str, default="0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    dirname = args.dirname
    start = args.start
    end = args.end

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])

    logger = logging.getLogger(__name__)
    logger.info("Start calculation")

    loop_over_directory = True
    if loop_over_directory:
        # directory = "/ABO3_cif_large/"
        # directory = "/ABO3_cif/"
        directory = "/ABO3_Mn_cif/"
        # directory = "/sugawara/"

        cif_files = glob.glob(os.getcwd() + directory + "*.cif")
        cif_files = sorted(cif_files)

        if end is None:
            end = len(cif_files)

        logger.info(f"Found {len(cif_files)} files, and do calculation from {start} to {end}.")

        materials = []
        etas = []

        for cif_file in cif_files[start:end]:

            if not os.path.isfile(cif_file):
                logger.info(f"Could not found file: {cif_file}")

            eta = get_overpotential_for_cif(cif_file=cif_file, dirname=dirname)

            material = os.path.basename(cif_file).split("_")[1].split(".")[0]

            if eta is not None:
                logger.info(f"material: {material:12.10s} eta: {eta:5.3f} eV")
                materials.append(material)
                etas.append(eta)
            else:
                logger.info(f"failed for {material}")

    else:
        cif_file = "CaMn2O4_ICSD_CollCode280514.cif"

        if not os.path.isfile(cif_file):
          logger.info(f"Could not found file: {cif_file}")

        eta = get_overpotential_for_cif(cif_file=cif_file, dirname=dirname)
        material = os.path.basename(cif_file).split("_")[1].split(".")[0]

        if eta is None:
            logger.info(f"failed for {material}")
        else:
            logger.info(f"file = {material:16.14s}, eta = {eta:5.3f} eV")

    make_barplot(labels=materials, values=etas, threshold=1000)
    
    logger.info("All done")
