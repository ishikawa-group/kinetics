def get_overpotential_for_cif():
    import numpy as np
    import argparse
    from kinetics.microkinetics.get_reaction_energy import get_reaction_energy
    from kinetics.microkinetics.utils import fix_lower_surface
    from ase.visualize import view
    from ase.build import fcc111

    parser = argparse.ArgumentParser()
    parser.add_argument("--unique_id", default="0")
    parser.add_argument("--replace_percent", default=0)
    args = parser.parse_args()
    unique_id = args.unique_id

    surface = fcc111(symbol="Ni", size=(2, 2, 3), vacuum=8.0)
    surface = fix_lower_surface(surface)

    energy_shift = [0] * 4

    reaction_file = "nh3_decomp.txt"

    # --- DFT calculation
    # try:
    deltaEs = get_reaction_energy(reaction_file=reaction_file, surface=surface, calculator="emt",
                                  verbose=True, dirname=unique_id)
    # except Exception as e:
    #    print("DFT calculation failed for this material")
    #    return None

    return deltaEs


if __name__ == "__main__":
    deltaEs = get_overpotential_for_cif()
    print("deltaEs:", deltaEs)
