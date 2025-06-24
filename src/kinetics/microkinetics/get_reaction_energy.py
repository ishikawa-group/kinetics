def register(db=None, atoms=None, data=None):
    formula = atoms.get_chemical_formula()

    db.write(atoms, name=formula, data=data)
    return None


def get_past_atoms(db=None, atoms=None):
    formula = atoms.get_chemical_formula()

    try:
        id_ = db.get(name=formula).id
        first = False
        atoms = db.get_atoms(id=id_).copy()
    except Exception as e:
        first = True

    finally:
        return atoms, first


def get_past_energy(db=None, atoms=None):
    formula = atoms.get_chemical_formula()
    try:
        id_ = db.get(name=formula).id
        first = False
        energy = db.get(id=id_).data.energy
    except Exception as e:
        first = True
        energy = None
    finally:
        return energy, first


def optimize_geometry(atoms=None, steps=30):
    import copy
    from ase.optimize import FIRE
    import logging
    import warnings
    warnings.filterwarnings("ignore")

    logger = logging.getLogger(__name__)

    atoms_ = copy.deepcopy(atoms)

    if "vasp" in atoms_.calc.name:
        try:
            atoms_.get_potential_energy()
        except Exception as e:
            logger.info("Error at VASP 1")
    else:
        name = atoms_.get_chemical_formula()
        trajectory = name + ".traj"
        opt = FIRE(atoms_, trajectory=trajectory, logfile="opt_log.txt")
        opt.run(fmax=0.05, steps=steps)

    return atoms_


def get_reaction_energy(reaction_file="oer.txt", surface=None, calculator="emt", input_yaml=None,
                        verbose=False, dirname="work"):
    """
    Calculate reaction energy for each reaction.
    """
    from pathlib import Path
    import numpy as np
    import copy
    import yaml
    import torch
    from ase.build import add_adsorbate
    from ase.db import connect
    from ase.visualize import view
    from ase.io import write
    from .utils import get_adsorbate_type
    from .utils import get_number_of_reaction
    from .utils import get_reac_and_prod
    from .vasp import set_vasp_calculator
    from .vasp import set_lmaxmix
    from ase.build import bulk
    import logging

    logger = logging.getLogger(__name__)
    np.set_printoptions(formatter={"float": "{:0.2f}".format})

    r_ads, r_site, r_coef, p_ads, p_site, p_coef = get_reac_and_prod(reaction_file)
    rxn_num = get_number_of_reaction(reaction_file)

    # load molecule collection
    package_dir = Path(__file__).resolve().parent
    database_path = package_dir / "data" / "g2plus.json"
    if database_path.exists():
        database = connect(database_path)
    else:
        logger.info(f"{database_path} not found.")
        raise FileNotFoundError

    # temporary database
    tmpdbfile = "tmp_" + dirname + ".db"
    tmpdb = connect(tmpdbfile)

    # reaction energy
    deltaEs = np.array([])

    # define calculator for molecules and surfaces separately
    calculator = calculator.lower()

    if "emt" in calculator:
        from ase.calculators.emt import EMT
        logger.info("EMT calculator is used.")
        calc_mol = EMT()
        calc_surf = EMT()

    elif "vasp" in calculator:
        # Load dfttype parameter from YAML file
        with open(input_yaml) as f:
            logger.info(f"Reading {input_yaml}")
            vasp_params = yaml.safe_load(f)
            dfttype = vasp_params["dfttype"]

        calc_mol = set_vasp_calculator(atom_type="molecule", input_yaml=input_yaml, do_optimization=True,
                                       dfttype=dfttype)
        calc_surf = set_vasp_calculator(atom_type="surface", input_yaml=input_yaml, do_optimization=True,
                                        dfttype=dfttype)

    elif "m3gnet" in calculator:
        import matgl
        from matgl.ext.ase import PESCalculator

        potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        calc_mol = PESCalculator(potential=potential)
        calc_surf = PESCalculator(potential=potential)

    elif "mattersim" in calculator:
        from mattersim.forcefield.potential import MatterSimCalculator

        device = "cuda" if torch.cuda.is_available() else "cpu"
        calc_mol = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)
        calc_surf = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)

    else:
        raise ValueError("Choose from emt, vasp, ocp.")

    # load adsorbate position from "adsorbate.yaml"
    with open("adsorbate.yaml") as f:
        ads_params = yaml.safe_load(f)
        offset = ads_params["offset"]
        height = ads_params["height"]

    # rotational angle for adsorbed molecules
    # note: also input duplicated name (e.g. "HO" and "OH")
    rotation = {"HO": [170, "x"], "OH": [170, "x"],
                "HO2": [70, "x"], "OOH": [70, "x"], "O2H": [70, "x"],
                "O2": [120, "x"],
                "HN": [180, "x"], "NH": [180, "x"],
                "H2N": [180, "x"], "NH2": [180, "x"],
                "H3N": [180, "x"], "NH3": [180, "x"],
                }

    # spin-polarized or not for adsorbed molecules
    closed_shell_molecules = ["H2",
                              "HO", "OH",
                              "H2O",
                              "N2",
                              "NH", "HN",
                              "NH2", "H2N",
                              "NH3", "H3N",
                              ]

    # magnetic elements -- magmom up for these elements
    magnetic_elements = ["Cr", "Mn", "Fe", "Co", "Ni"]

    # zero-point energy (in cm^-1, NIST webbook, experimental)
    add_zpe_here = False
    zpe = {}
    cm_to_eV = 1.23984e-4
    zpe_cm = {"H2": 2179.307,
              "HO": 1850.688, "OH": 1850.688,
              "H2O": 4504.0,
              "HO2": 2962.8, "OOH": 2962.8, "O2H": 2962.8,
              "O2": 787.3797,
              "N2": 1175.778,
              "HN": 1623.563, "NH": 1623.563,
              "H2N": 4008.9, "NH2": 4008.9,
              "H3N": 7214.5, "NH3": 7214.5,
              }
    for key, value in zpe_cm.items():
        zpe.update({key: value * cm_to_eV})

    for irxn in range(rxn_num):
        energies = {"reactant": 0.0, "product": 0.0}

        for side in ["reactant", "product"]:
            surf_ = copy.deepcopy(surface)

            # assume high spin for surface
            symbols = surf_.get_chemical_symbols()
            init_magmom = [1.0 if x in magnetic_elements else 0.0 for x in symbols]
            surf_.set_initial_magnetic_moments(init_magmom)

            if side == "reactant":
                mols, sites, coefs = r_ads[irxn], r_site[irxn], r_coef[irxn]
            elif side == "product":
                mols, sites, coefs = p_ads[irxn], p_site[irxn], p_coef[irxn]
            else:
                logger.info("some error")
                quit()

            E = 0.0

            for imol, mol in enumerate(mols):
                if mol[0] == "surf":
                    atoms = surf_
                else:
                    try:
                        id_ = database.get(name=mol[0]).id
                    except KeyError:
                        logger.info(f"{mol[0]} not found in {database_file}")
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

                    position = adsorbate.positions[0][:2]

                    # check whether the bare surface is calcualted before
                    surf_, first = get_past_atoms(db=tmpdb, atoms=surf_)

                    if first:
                        formula = surf_.get_chemical_formula()
                        work_dir = Path(dirname) / formula
                        work_dir.mkdir(parents=True, exist_ok=True)
                        surf_.calc = calc_surf
                        surf_.calc.directory = str(work_dir)

                        if "vasp" in calculator and "plus" in dfttype:
                            set_lmaxmix(atoms=surf_)

                        surf_ = optimize_geometry(atoms=surf_)
                        register(db=tmpdb, atoms=surf_, data={"energy": 0.0})
                    else:
                        # Not the first time for - use previous value
                        pass

                    atoms = copy.deepcopy(surf_)
                    add_adsorbate(atoms, adsorbate, offset=offset, position=position, height=height)
                    atoms.calc = calc_surf
                else:
                    logger.info("some error")
                    quit()

                # setting atoms done
                energy, first = get_past_energy(db=tmpdb, atoms=atoms)
                formula = atoms.get_chemical_formula()

                if first:
                    if verbose:
                        logger.info(f"Calculating {formula}")

                    work_dir = Path(dirname) / formula
                    work_dir.mkdir(parents=True, exist_ok=True)
                    pngfile = work_dir / Path(atoms.get_chemical_formula() + ".png")
                    write(pngfile, atoms)

                    atoms.calc.directory = str(work_dir)

                    if "vasp" in calculator and "plus" in dfttype:
                        set_lmaxmix(atoms=atoms)

                    try:
                        energy = atoms.get_potential_energy()
                    except Exception:
                        logger.info("Energy calculation error")
                        raise ValueError

                    try:
                        register(db=tmpdb, atoms=atoms, data={"energy": energy})
                    except Exception:
                        logger.warning(f"Failed to write to {tmpdb}")

                # add zpe for gaseous molecule
                if add_zpe_here:
                    if ads_type == "gaseous" and mol[0] != "surf":
                        zpe_value = zpe[mol[0]]
                        energy += zpe_value

                E += coefs[imol] * energy

            energies[side] = E

        deltaE = energies["product"] - energies["reactant"]
        deltaEs = np.append(deltaEs, deltaE)

    # loop over reaction - done
    logger.info(f"deltaEs = {deltaEs}")

    return deltaEs
