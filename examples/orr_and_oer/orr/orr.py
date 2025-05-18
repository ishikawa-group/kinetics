import uuid
import sys
import os
import glob
import argparse
import logging
from kinetics.microkinetics.orr_and_oer import get_overpotential_for_cif
from kinetics.microkinetics.utils import make_barplot

if __name__ == "__main__":
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

    reaction_file = "orr_alkaline2.txt"
    energy_shift = [-0.32, -0.54, -0.47, -0.75]
    calculator = "m3gnet"

    directory = None
    # directory = "/ABO3_cif_large/"
    # directory = "/ABO3_cif/"
    # directory = "/ABO3_Mn_cif/"

    if directory is not None:  # Directory mode
        cif_files = sorted(glob.glob(os.getcwd() + directory + "*.cif"))

        for cif_file in cif_files:
            if not os.path.isfile(cif_file):
                logger.info(f"Could not found file: {cif_file}")
                continue

            material = os.path.basename(cif_file).split("_")[1].split(".")[0]
            eta = get_overpotential_for_cif(cif_file=cif_file, calculator=calculator,
                                            reaction_type="orr", reaction_file=reaction_file)

            if eta is not None:
                logger.info(f"material: {material:12.10s} eta: {eta:5.3f} eV")
                materials.append(material)
                etas.append(eta)
            else:
                logger.info(f"failed for {material}")

        make_barplot(labels=materials, values=etas, threshold=1000)

    else:  # Single file mode
        cif_file = "CaMn2O4_ICSD_CollCode280514.cif"
        if not os.path.isfile(cif_file):
            raise ValueError(f"Could not found file: {cif_file}")

        logger.info(f"Calculating {cif_file}")
        material = os.path.basename(cif_file).split("_")[1].split(".")[0]

        eta = get_overpotential_for_cif(cif_file=cif_file, calculator=calculator,
                                        reaction_type="orr", reaction_file=reaction_file)

        if eta is None:
            logger.info(f"failed for {material}")
        else:
            logger.info(f"file = {material:16.14s}, eta = {eta:5.3f} eV")
            materials.append(material)
            etas.append(eta)

    logger.info("All done")
