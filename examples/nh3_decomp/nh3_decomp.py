def make_energy_diagram(deltaEs=None, savefig=True, figname="ped.png", xticklabels=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import interpolate

    deltaEs = np.array(deltaEs)
    num_rxn = len(deltaEs)

    ped = np.zeros(1)  # 0)
    for i in range(num_rxn):
        ped = np.append(ped, ped[-1] + deltaEs[i])

    points = 500  # resolution

    y1 = ped

    has_barrier = False
    if has_barrier:
        rds = 1
        alpha = 0.87
        beta = 1.34
        Ea  = y1[rds] * alpha + beta  # BEP

        # extend x length after TS curve
        y1  = np.insert(y1, rds, y1[rds])
        num_rxn += 1

    x1_latent = np.linspace(-0.5, num_rxn + 0.5, points)
    x1 = np.arange(0, num_rxn+1)
    f1 = interpolate.interp1d(x1, y1, kind="nearest", fill_value="extrapolate")
    #
    # replace RDS by quadratic curve
    #
    if has_barrier:
        x2 = [rds - 0.5, rds, rds + 0.5]
        x2 = np.array(x2)
        y2 = np.array([y1[rds-1], Ea, y1[rds+1]])
        f2 = interpolate.interp1d(x2, y2, kind="quadratic")

    y = np.array([])
    for i in x1_latent:
        val1 = f1(i)
        val2 = -1.0e10
        if has_barrier:
            val2 = f2(i)
        y = np.append(y, max(val1, val2))

    # when saving png file
    if savefig:
        sns.set(style="darkgrid", rc={"lines.linewidth": 2.0, "figure.figsize": (10, 4)})
        p = sns.lineplot(x=x1_latent, y=y, sizes=(0.5, 1.0))
        p.set_xlabel("Steps", fontsize=16)
        p.set_ylabel("Energy (eV)", fontsize=16)
        p.tick_params(axis='both', labelsize=14)
        p.yaxis.set_major_formatter(lambda x, p: f"{x:.1f}")
        if xticklabels is not None:
            xticklabels.insert(0, "dummy")
            p.set_xticklabels(xticklabels, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(figname)

    return None


if __name__ == "__main__":
    import logging
    from kinetics.microkinetics.utils import fix_lower_surface, sort_atoms_by_z
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

    reaction_file = "nh3_decomp.txt"; energy_shift = [0] * 4

    deltaEs = get_reaction_energy(reaction_file=reaction_file, surface=surface,
                                  calculator=calculator, verbose=True, dirname="work")

    print("deltaEs:", np.round(np.array(deltaEs), 3))

    xticklabels = ["NH3 + *", "NH3*", "NH2* + H*", "NH* + H*", "N* + H*", "N2 + H*", "N2 + H2"]
    make_energy_diagram(deltaEs=deltaEs, xticklabels=xticklabels)
