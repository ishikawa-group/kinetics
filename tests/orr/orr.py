def get_overpotential_for_cif(cif_file=None):
    import sys
    sys.path.append("../../")

    import numpy as np
    import argparse
    from kinetics.microkinetics.utils import make_surface_from_cif
    from kinetics.microkinetics.utils import remove_layers
    from kinetics.microkinetics.utils import replace_element
    from kinetics.microkinetics.utils import fix_lower_surface
    from kinetics.microkinetics.utils import sort_atoms_by_z
    from kinetics.microkinetics.get_reaction_energy import get_reaction_energy
    from kinetics.microkinetics.orr_and_oer import get_overpotential_oer_orr 
    from ase.visualize import view

    parser = argparse.ArgumentParser()
    parser.add_argument("--unique_id", default="0")
    parser.add_argument("--replace_percent", default=0)
    args = parser.parse_args()
    unique_id = args.unique_id
    replace_percent = int(args.replace_percent)

    repeat = [2, 1, 1]

    vacuum = 8.0
    surface = make_surface_from_cif(cif_file, indices=[0, 0, 1], repeat=repeat, vacuum=vacuum)

    # surface, count = sort_atoms_by_z(surface)
    # lowest_z = surface[0].position[2]
    # surface.translate([0, 0, -lowest_z + 0.5])

    # surface = remove_layers(surface, element="Ca", layers_to_remove=1)
    # surface = remove_layers(surface, element="O", layers_to_remove=1)
    # surface = replace_element(surface, from_element="Mn", to_element="Cr", percent=100)
    surface = fix_lower_surface(surface)

    energy_shift = [0]*4

    # reaction_file = "orr_alkaline.txt"  # not really good on first step
    # reaction_file = "orr_alkaline2.txt"; energy_shift = [-0.32+0.75, -0.54+0.32, -0.47+0.54, -0.75+0.47]
    reaction_file = "orr_alkaline3.txt"; energy_shift = [-0.32+0.75-4.92, -0.54+0.32, -0.47+0.54, -0.75+0.47]
    # reaction_file = "orr_alkaline3.txt"; energy_shift = [-4.92, 0, 0, 0]

    # --- DFT calculation
    deltaEs = get_reaction_energy(reaction_file=reaction_file, surface=surface, calculator="vasp",
                                  verbose=True, dirname=unique_id)

    eta = get_overpotential_oer_orr(reaction_file=reaction_file, deltaEs=deltaEs, reaction_type="orr",
                                    verbose=True, energy_shift=energy_shift)
    eta = np.abs(eta)

    return eta


if __name__ == "__main__":
    import os
    import glob
    from logging import basicConfig, getLogger, INFO

    basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = getLogger(__name__)

    loop_over_directory = False

    logger.info("Start calculation")

    if loop_over_directory:
        #
        # Loop over files in the directory
        #

        # directory = "/ABO3_cif_large/"
        directory = "/sugawara/"

        cif_files = glob.glob(os.getcwd() + directory + "*.cif")
        logger.info(f"Found {len(cif_files)} files.")

        for cif_file in cif_files:
            if not os.path.isfile(cif_file):
                logger.info(f"Could not found file: {cif_file}")

            eta = get_overpotential_for_cif(cif_file=cif_file)

            basename = os.path.basename(cif_file)
            if eta is None:
                logger.info(f"failed for {basename}")
            else:
                logger.info(f"file = {basename:26.24s}, eta = {eta:5.3f} eV")
    else:
        #
        # single file
        #

        cif_file = "CaMn2O4_ICSD_CollCode280514.cif"
        # cif_file = "ICSD_CollCode35218_CaMnO3.cif"

        if not os.path.isfile(cif_file):
            logger.info(f"Could not found file: {cif_file}")

        eta = get_overpotential_for_cif(cif_file=cif_file)

        basename = os.path.basename(cif_file)
        if eta is None:
            logger.info(f"failed for {basename}")
        else:
            logger.info(f"file = {basename:26.24s}, eta = {eta:5.3f} eV")

