def get_overpotential_for_cif(cif_file=None):
    import numpy as np
    import argparse
    from microkinetics_toolkit.utils import make_surface_from_cif
    from microkinetics_toolkit.utils import remove_layers
    from microkinetics_toolkit.utils import replace_element
    from microkinetics_toolkit.utils import fix_lower_surface
    from microkinetics_toolkit.get_reaction_energy import get_reaction_energy
    from microkinetics_toolkit.orr_and_oer import get_overpotential_oer_orr 
    from ase.visualize import view

    parser = argparse.ArgumentParser()
    parser.add_argument("--unique_id", default="0")
    parser.add_argument("--replace_percent", default=0)
    args = parser.parse_args()
    unique_id = args.unique_id
    replace_percent = int(args.replace_percent)

    # cif_file = "mp-1183920_CsPaO3.cif"
    # cif_file = "LaMnO3.cif"

    repeat = [1, 1, 2]
    # repeat = [2, 2, 2]

    vacuum = 8.0
    surface = make_surface_from_cif(cif_file, indices=[0, 0, 1], repeat=repeat, vacuum=vacuum)
    surface.translate([0, 0, -vacuum+0.1])

    # surface = remove_layers(surface, element="La", n_layers=4)
    # surface = remove_layers(surface, element="Mn", n_layers=3)
    # surface = remove_layers(surface, element="O", n_layers=11)

    # --- make random replacement
    # surface = replace_element(surface, from_element="Mn", to_element="Cr", percent=100)

    surface = fix_lower_surface(surface)

    energy_shift = [0]*4

    # reaction_file = "orr_alkaline.txt"  # not really good on first step
    reaction_file = "orr_alkaline2.txt"; energy_shift = [-0.32+0.75, -0.54+0.32, -0.47+0.54, -0.75+0.47]
    # reaction_file = "orr_alkaline3.txt"; energy_shift = [-0.32+0.75-4.92, -0.54+0.32, -0.47+0.54, -0.75+0.47]

    # --- DFT calculation
    try:
        deltaEs = get_reaction_energy(reaction_file=reaction_file, surface=surface, calculator="vasp",
                                      verbose=False, dirname=unique_id)
    except:
        print("DFT calculation failed for this material")
        return None

    eta = get_overpotential_oer_orr(reaction_file=reaction_file, deltaEs=deltaEs, reaction_type="orr",
                                    verbose=False, energy_shift=energy_shift)
    eta = np.abs(eta)

    return eta


if __name__ == "__main__":
    import os
    import glob

    directory = '/ABO3_cif_large/'

    # Loop over files in the directory
    cif_files = glob.glob(os.getcwd() + directory + "*.cif")
    print(f"Found {len(cif_files)} files.", flush=True)

    for cif_file in cif_files:
        if not os.path.isfile(cif_file):
            print(f"Could not found file: {cif_file}")

        eta = get_overpotential_for_cif(cif_file=cif_file)

        basename = os.path.basename(cif_file)
        if eta is None:
            print(f"failed for {basename}")
        else:
            print(f"file = {basename:26.24s}, eta = {eta:5.3f} eV", flush=True)
