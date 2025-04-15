if __name__ == "__main__":
    import logging
    from kinetics.microkinetics.utils import fix_lower_surface, sort_atoms_by_z, make_energy_diagram
    from kinetics.microkinetics.get_reaction_energy import get_reaction_energy
    import numpy as np
    from ase.build import fcc111

    logger = logging.getLogger(__name__)

    vacuum = 6.5
    surface = fcc111(symbol="Ni", size=(3, 3, 4), vacuum=vacuum, periodic=True)
    surface, count = sort_atoms_by_z(surface)
    lowest_z = surface[0].position[2]
    surface.translate([0, 0, -vacuum+0.1])
    surface = fix_lower_surface(surface)

    calculator = "m3gnet"

    reaction_file = "nh3_decomp.txt"
    energy_shift = [0] * 4

    deltaEs = get_reaction_energy(reaction_file=reaction_file, surface=surface,
                                  calculator=calculator, verbose=True, dirname="work")

    print("deltaEs:", np.round(np.array(deltaEs), 3))

    xticklabels = ["NH3 + *", "NH3*", "NH2* + H*", "NH* + H*", "N* + H*", "N2 + H*", "N2 + H2"]
    make_energy_diagram(deltaEs=deltaEs, xticklabels=xticklabels)
