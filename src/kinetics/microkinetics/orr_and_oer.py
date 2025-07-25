from kinetics.utils import sort_atoms_by_z, fix_lower_surface
from kinetics.microkinetics.get_reaction_energy import get_reaction_energy
from ase import Atoms
from ase.visualize import view


def get_overpotential_oer_orr(reaction_file, deltaEs, T=298.15, reaction_type="oer",
                              energy_shift=None, verbose=False):
    """
    Calculate overpotential for OER or ORR.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from kinetics.utils import get_number_of_reaction
    import logging

    logger = logging.getLogger(__name__)

    np.set_printoptions(formatter={"float": "{:0.2f}".format})

    rxn_num = get_number_of_reaction(reaction_file)

    # check the contents of deltaE
    if any(e is None for e in deltaEs):
        return None

    zpe = {"H2": 0.0, "H2O": 0.0, "OHads": 0.0, "Oads": 0.0, "OOHads": 0.0}
    S = {"H2": 0.0, "H2O": 0.0, "O2": 0.0}

    # ZPE in eV
    add_zpe_here = True
    if add_zpe_here:
        zpe["H2"] = 0.27
        zpe["H2O"] = 0.56
        zpe["OHads"] = 0.36
        zpe["Oads"] = 0.07
        zpe["OOHads"] = 0.40
        zpe["O2"] = 0.05 * 2

    # entropy in eV/K
    S["H2"] = 0.41 / T
    S["H2O"] = 0.67 / T
    S["O2"] = 0.32 * 2 / T

    # loss in entropy in each reaction
    deltaSs = np.zeros(rxn_num)
    deltaZPEs = np.zeros(rxn_num)

    reaction_type = reaction_type.lower()
    if reaction_type == "oer":
        deltaSs[0] = 0.5 * S["H2"] - S["H2O"]
        deltaSs[1] = 0.5 * S["H2"]
        deltaSs[2] = 0.5 * S["H2"] - S["H2O"]
        deltaSs[3] = 2.0 * S["H2O"] - 1.5 * S["H2"]

        deltaZPEs[0] = zpe["OHads"] + 0.5 * zpe["H2"] - zpe["H2O"]
        deltaZPEs[1] = zpe["Oads"] + 0.5 * zpe["H2"] - zpe["OHads"]
        deltaZPEs[2] = zpe["OOHads"] + 0.5 * zpe["H2"] - zpe["Oads"] - zpe["H2O"]
        deltaZPEs[3] = 2.0 * zpe["H2O"] - 1.5 * zpe["H2"] - zpe["OOHads"]

    elif reaction_type == "orr":
        deltaSs[0] = - S["O2"] - S["H2"]
        deltaSs[1] = S["H2O"] - 0.5 * S["H2"]
        deltaSs[2] = - 0.5 * S["H2"]
        deltaSs[3] = S["H2O"] - 0.5 * S["H2"]

        deltaZPEs[0] = zpe["OOHads"] - 0.5 * zpe["H2"] - zpe["O2"]
        deltaZPEs[1] = zpe["Oads"] + zpe["H2O"] - 0.5 * zpe["H2"] - zpe["OOHads"]
        deltaZPEs[2] = zpe["OHads"] - 0.5 * zpe["H2"] - zpe["Oads"]
        deltaZPEs[3] = zpe["H2O"] - 0.5 * zpe["H2"] - zpe["OHads"]

    else:
        raise ValueError("Error at orr_and_oer.py")

    deltaEs = np.array(deltaEs)
    deltaHs = deltaEs + deltaZPEs
    deltaGs = deltaHs - T * deltaSs

    if energy_shift is not None:
        deltaGs += np.array(energy_shift)

    if verbose:
        logger.info(f"max of deltaGs = {np.max(deltaGs):5.3f} eV")

    if reaction_type == "orr":
        # phi = 1.165  # equilibrium potential, 4.661/4, from Wang's paper
        phi = 1.0288
        deltaGs_sum = [0.0,
                       deltaGs[0],
                       deltaGs[0] + deltaGs[1],
                       deltaGs[0] + deltaGs[1] + deltaGs[2],
                       deltaGs[0] + deltaGs[1] + deltaGs[2] + deltaGs[3]]

        deltaGs_eq = [deltaGs_sum[0],
                      deltaGs_sum[1] + phi,
                      deltaGs_sum[2] + 2 * phi,
                      deltaGs_sum[3] + 3 * phi,
                      deltaGs_sum[4] + 4 * phi]

        diffG = [deltaGs_eq[1] - deltaGs_eq[0], deltaGs_eq[2] - deltaGs_eq[1],
                 deltaGs_eq[3] - deltaGs_eq[2], deltaGs_eq[4] - deltaGs_eq[3]]

        eta = np.max(diffG)

    elif reaction_type == "oer":
        phi = 1.23
        deltaGs_sum = [0.0,
                       deltaGs[0],
                       deltaGs[0] + deltaGs[1],
                       deltaGs[0] + deltaGs[1] + deltaGs[2],
                       deltaGs[0] + deltaGs[1] + deltaGs[2] + deltaGs[3]]

        deltaGs_eq = [deltaGs_sum[0],
                      deltaGs_sum[1] + phi,
                      deltaGs_sum[2] + 2 * phi,
                      deltaGs_sum[3] + 3 * phi,
                      deltaGs_sum[4] + 4 * phi]

        diffG = [deltaGs_eq[1] - deltaGs_eq[0], deltaGs_eq[2] - deltaGs_eq[1],
                 deltaGs_eq[3] - deltaGs_eq[2], deltaGs_eq[4] - deltaGs_eq[3]]

        eta = np.max(diffG)

    if verbose:
        logger.info(f"deltaGs = {np.array(deltaGs)}")
        logger.info(f"deltaGs_eq = {np.array(deltaGs_eq)}")
        logger.info(f"diffG = {np.array(diffG)}")
        eta = np.max(diffG)

    # plot
    fig_name = "test.png"
    plt.plot(deltaGs_eq, "o")
    plt.savefig(fig_name)

    eta = np.abs(eta)
    return eta


def get_overpotential_for_atoms(
        surface: Atoms=None,
        reaction_file=None,
        energy_shift=None,
        calculator="mace",
        input_yaml="vasp_default.yaml",
        reaction_type="orr") -> float:

    surface, count = sort_atoms_by_z(surface)
    lowest_z = surface[0].position[2]
    surface.translate([0, 0, -lowest_z + 0.1])

    surface = fix_lower_surface(surface)
    deltaEs = get_reaction_energy(reaction_file=reaction_file, surface=surface, calculator=calculator,
                                  input_yaml=input_yaml)
    eta = get_overpotential_oer_orr(reaction_file=reaction_file, deltaEs=deltaEs,
                                    reaction_type=reaction_type, energy_shift=energy_shift)
    return eta
