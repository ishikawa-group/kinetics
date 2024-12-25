if __name__ == "__main__":
    import numpy as np
    from kinetics.microkinetics.get_reaction_energy import get_reaction_energy
    from kinetics.microkinetics.utils import fix_lower_surface
    from ase.visualize import view
    from ase.build import fcc111

    surface = fcc111(symbol="Ni", size=(2, 2, 3), vacuum=8.0)
    surface = fix_lower_surface(surface)

    energy_shift = [0] * 4

    reaction_file = "nh3_decomp.txt"

    deltaEs = get_reaction_energy(reaction_file=reaction_file, surface=surface, calculator="emt", verbose=True)

    print("deltaEs:", deltaEs)
