def register(db=None, atoms=None, formula=None, data=None):
    formula = atoms.get_chemical_formula()
    db.write(atoms, name=formula, data=data)
    return None


def get_past_atoms(db=None, atoms=None):
    formula = atoms.get_chemical_formula()
    try:
        id_ = db.get(name=formula).id
        first = False
        atoms = db.get_atoms(id=id_).copy()
    except:
        first = True
        atoms = atoms
    finally:
        return atoms, first


def get_past_energy(db=None, atoms=None):
    return atoms_, first


def get_reaction_energy(reaction_file="oer.txt", surface=None, calculator="emt", verbose=False, dirname=None):
    """
    Calculate reaction energy for each reaction.
    """
    import os
    import numpy as np
    from ase.build import add_adsorbate
    from ase.calculators.emt import EMT
    from ase.db import connect
    from ase.visualize import view
    from microkinetics_toolkit.utils import get_adsorbate_type
    from microkinetics_toolkit.utils import get_number_of_reaction
    from microkinetics_toolkit.utils import get_reac_and_prod
    from microkinetics_toolkit.vasp import set_vasp_calculator
    from microkinetics_toolkit.vasp import set_lmaxmix

    r_ads, r_site, r_coef, p_ads, p_site, p_coef = get_reac_and_prod(reaction_file)
    rxn_num = get_number_of_reaction(reaction_file)

    # load molecule collection
    database_file = "data/g2plus.json"
    if os.path.exists(database_file):
        database = connect(database_file)
    else:
        raise FileNotFoundError

    # temporary database
    tmpdbfile = "tmp_" + dirname + ".db"
    tmpdb = connect(tmpdbfile)

    # reaction energy
    deltaEs = np.array([])

    # define calculator for molecules and surfaces separately
    if "emt" in calculator:
        calc_mol  = EMT()
        calc_surf = EMT()
    elif "vasp" in calculator:
        kpt = 1
        dfttype = "gga"  # "plus_u"
        calc_mol  = set_vasp_calculator(atom_type="molecule", do_optimization=True, dfttype=dfttype, kpt=kpt)
        calc_surf = set_vasp_calculator(atom_type="surface", do_optimization=True, dfttype=dfttype, kpt=kpt)
    elif "ocp" in valculator:
        calc_mol  = set_ocp_calculator()  # do not work
        calc_surf = set_ocp_calculator()
    else:
        raise ValueError("Choose from emt, vasp, ocp.")

    # rotational angle for adsorbed molecules
    # rotation = {"HO": [180, "x"], "HO2": [180, "x"], "O2": [90, "x"]}
    rotation = {"HO": [160, "x"], "HO2": [160, "x"], "O2": [70, "x"]}

    # spin-polarized or not for adsorbed molecules
    closed_shell_molecules = ["H2", "HO", "H2O"]

    # magnetic elements: B in ABO3 perovskite
    magnetic_elements = ["Mn", "Fe", "Cr"]

    for irxn in range(rxn_num):
        energies = {"reactant": 0.0, "product": 0.0}

        for side in ["reactant", "product"]:
            surf_ = surface.copy()

            # assume high spin for surface
            symbols = surf_.get_chemical_symbols()
            init_magmom = [1.0 if x in magnetic_elements else 0.0 for x in symbols]
            surf_.set_initial_magnetic_moments(init_magmom)

            if side == "reactant":
                mols, sites, coefs = r_ads[irxn], r_site[irxn], r_coef[irxn]
            elif side == "product":
                mols, sites, coefs = p_ads[irxn], p_site[irxn], p_coef[irxn]
            else:
                print("some error")
                quit()

            E = 0.0

            for imol, mol in enumerate(mols):
                if mol[0] == "surf":
                    atoms = surf_
                else:
                    try:
                        id_ = database.get(name=mol[0]).id
                    except KeyError:
                        print(f"{mol[0]} not found in {database_file}", flush=True)
                        quit()
                    else:
                        atoms = database.get_atoms(id=id_)

                site = sites[imol][0]
                ads_type = get_adsorbate_type(atoms, site)

                if ads_type == "gaseous":
                    if mol[0] == "surf":
                        atoms.calc = calc_surf
                    else:
                        atoms.calc = calc_mol
                        atoms.cell = [20, 20, 20]
                        atoms.pbc = True
                        atoms.center()
                        if mol[0] in closed_shell_molecules:
                            atoms.calc.set(ispin=1)

                elif ads_type == "surface":
                    atoms.calc = calc_surf

                elif ads_type == "adsorbed":
                    adsorbate = atoms
                    tmp = adsorbate.get_chemical_formula()

                    if tmp in rotation:
                        adsorbate.rotate(*rotation[tmp])

                    height = 1.8
                    # offset = (0.0, 0.25)  # for middle cell
                    offset = (0.0, 0.50)  # for smallest cell
                    position = adsorbate.positions[0][:2]

                    # check whether the bare surface is calcualted before
                    surf_, first = get_past_atoms(db=tmpdb, atoms=surf_)
                    if first:
                        print(f"First time to calculate bare surface.", flush=True)
                        formula = surf_.get_chemical_formula()
                        directory = "work_" + dirname + "/" + formula
                        surf_.calc = calc_surf
                        surf_.calc.directory = directory
                        set_lmaxmix(atoms=surf_)
                        surf_.get_potential_energy()
                        register(db=tmpdb, atoms=surf_)
                    else:
                        print(f"Bare surface found in database.", flush=True)

                    atoms = surf_.copy()
                    add_adsorbate(atoms, adsorbate, offset=offset, position=position, height=height)
                    atoms.calc = calc_surf
                else:
                    print("some error")
                    quit()

                # Setting atoms done. Perform energy calculation.
                formula = atoms.get_chemical_formula()
                directory = "work_" + dirname + "/" + formula
                atoms.calc.directory = directory
                set_lmaxmix(atoms=atoms)
                energy = atoms.get_potential_energy()

                E += coefs[imol]*energy

            energies[side] = E

        deltaE  = energies["product"] - energies["reactant"]
        deltaEs = np.append(deltaEs, deltaE)

    return deltaEs
