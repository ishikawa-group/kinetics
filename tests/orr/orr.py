def add_data_to_jsonfile(jsonfile, data):
    """
    add data to database
    """
    import json

    if not os.path.exists(jsonfile):
        with open(jsonfile, "w") as f:
            json.dump([], f)

    with open(jsonfile, "r") as f:
        datum = json.load(f)

        # remove "doing" record as calculation is done
        for i in range(len(datum)):
            if datum[i]["status"] == "doing":
                datum.pop(i)
                break

        datum.append(data)

    with open(jsonfile, "w") as f:
        json.dump(datum, f, indent=4)


def get_overpotential_for_cif(cif_file=None, dirname=None):
    import sys
    sys.path.append("../../")

    import numpy as np
    import json
    import uuid
    import argparse
    from kinetics.microkinetics.utils import make_surface_from_cif
    from kinetics.microkinetics.utils import remove_layers
    from kinetics.microkinetics.utils import replace_element
    from kinetics.microkinetics.utils import fix_lower_surface
    from kinetics.microkinetics.utils import sort_atoms_by_z
    from kinetics.microkinetics.get_reaction_energy import get_reaction_energy
    from kinetics.microkinetics.orr_and_oer import get_overpotential_oer_orr 
    from ase.visualize import view

    # replace_percent = 0

    repeat = [1, 1, 2]

    vacuum = 6.3
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
    # reaction_file = "orr_alkaline2.txt"; energy_shift = [-0.32+0.75, -0.54+0.32, -0.47+0.54, -0.75+0.47]
    # reaction_file = "orr_alkaline3.txt"; energy_shift = [-0.32+0.75-4.92, -0.54+0.32, -0.47+0.54, -0.75+0.47]
    reaction_file = "orr_alkaline3.txt"; energy_shift = [-4.92, 0, 0, 0]

    # --- DFT calculation
    calculator = "vasp"
    dfttype = "plus_u"  # ( "gga" | "plus_u" | "meta_gga" )

    try:
        deltaEs = get_reaction_energy(reaction_file=reaction_file, surface=surface, calculator=calculator, dfttype=dfttype, verbose=True, dirname=dirname)
    except:
        logger.info("Error: failed for some reason")
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


def make_barplot(labels=None, values=None):
    import matplotlib.pyplot as plt
    import numpy as np

    sorted_indices = np.argsort(values)
    sorted_labels  = [labels[i] for i in sorted_indices]
    sorted_values  = [values[i] for i in sorted_indices]

    plt.figure(figsize=(8,5))
    plt.bar(sorted_labels, sorted_values, color="skyblue")

    plt.ylabel("Overpotential (eV)")
    plt.savefig("bar_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    import os
    import glob
    from ase.io import read
    from logging import basicConfig, getLogger, INFO
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", type=str, default="0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    dirname = args.dirname
    start = args.start
    end = args.end

    basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = getLogger(__name__)

    loop_over_directory = True

    logger.info("Start calculation")

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

        cif_files = cif_files[start:end]

        materials = []
        etas = []
        for cif_file in cif_files:

            if not os.path.isfile(cif_file):
                logger.info(f"Could not found file: {cif_file}")

            eta = get_overpotential_for_cif(cif_file=cif_file, dirname=dirname)

            basename = os.path.basename(cif_file)
            material = basename.split("_")[1].split(".")[0]

            if eta is not None:
                logger.info(f"file = {material:22.20s}, eta = {eta:5.3f} eV")
                materials.append(material)
                etas.append(eta)
            else:
                logger.info(f"failed for {basename}")

        make_barplot(labels=materials, values=etas)

    else:
        cif_file = "CaMn2O4_ICSD_CollCode280514.cif"
        # cif_file = "ICSD_CollCode35218_CaMnO3.cif"

        if not os.path.isfile(cif_file): logger.info(f"Could not found file: {cif_file}")

        eta = get_overpotential_for_cif(cif_file=cif_file)

        basename = os.path.basename(cif_file)
        if eta is None:
            logger.info(f"failed for {basename}")
        else:
            logger.info(f"file = {basename:26.24s}, eta = {eta:5.3f} eV")

    logger.info("All done")

