if __name__ == "__main__":
    import sys
    sys.path.append("../../")

    import numpy as np
    from kinetics.microkinetics.get_reaction_energy import get_reaction_energy
    from kinetics.microkinetics.utils import fix_lower_surface
    from ase.visualize import view
    from ase.build import fcc111

    vacuum = 6.0

    surface = fcc111(symbol="Ni", size=(2, 2, 4), vacuum=vacuum, periodic=True)

    surface.translate([0, 0, -vacuum+0.1])
    surface = fix_lower_surface(surface)

    energy_shift = [0] * 4

    reaction_file = "nh3_decomp.txt"

    deltaEs = get_reaction_energy(reaction_file=reaction_file, surface=surface, calculator="vasp", verbose=True)

    print("deltaEs:", deltaEs)
