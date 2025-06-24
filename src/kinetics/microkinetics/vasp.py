def set_vasp_calculator(atom_type="molecule", input_yaml="vaspinput_template.yaml", do_optimization=False, dfttype=None):
    """
    Set up a calculator using VASP.

    Args:
        atom_type: "molecule", "surface" or "solid".
        do_optimization: True or False.
    """
    import json
    import yaml
    from pathlib import Path
    from ase.calculators.vasp import Vasp

    # Load VASP parameters from YAML file
    with open(input_yaml) as f:
        vasp_params = yaml.safe_load(f)

    # Get atom type specific settings
    if atom_type not in vasp_params:
        raise ValueError(f"Invalid atom_type: {atom_type}")
    
    type_params = vasp_params[atom_type]
    common_params = vasp_params["common"]
    mixing_params = vasp_params["mixing"]
    
    # DFT + U
    if "plus" in dfttype:
        dft_u_params = vasp_params["dft_plus_u"]
        ldau = dft_u_params["ldau"]
        lasph = dft_u_params["lasph"]
        ldautype = dft_u_params["ldautype"]

        # Load U parameters from JSON file
        u_param_file = dft_u_params["u_param_file"]
        with open(u_param_file) as f:
            ldau_luj = json.load(f)
    else:
        ldau = None
        ldautype = None
        ldau_luj = None
        lasph = common_params["lasph"]

    # metagga
    if "meta" in dfttype:
        common_params["xc"] = "r2scan"
        lasph = True
        common_params["algo"] = "Normal"

    # geometry optimization related
    opt_params = vasp_params["optimization"]["enabled" if do_optimization else "disabled"]

    # Combine all parameters
    calc_params = {
        # Atom type specific parameters
        "kpts": type_params["kpts"],
        "ismear": type_params["ismear"],
        "sigma": type_params["sigma"],
        "lreal": type_params["lreal"],
        "ldipol": type_params["ldipol"],
        "idipol": type_params["idipol"],

        # Common parameters
        "prec": "Normal",
        "xc": common_params["xc"],
        "pp": "pbe",
        "encut": common_params["encut"],
        "ediff": common_params["ediff"],
        "ediffg": common_params["ediffg"],
        "lorbit": common_params["lorbit"],
        "algo": common_params["algo"],
        "nelm": common_params["nelm"],
        "nelmin": common_params["nelmin"],
        "npar": common_params["npar"],
        "nsim": common_params["nsim"],
        "ispin": common_params["ispin"],
        "isym": common_params["isym"],
        "kgamma": common_params["kgamma"],
        "lasph": lasph,
        "lwave": common_params["lwave"],
        "lcharg": common_params["lcharg"],
        "kspacing": common_params["kspacing"],

        # Mixing parameters
        "amix": mixing_params["amix"],
        "amix_mag": mixing_params["amix_mag"],
        "bmix": mixing_params["bmix"],
        "bmix_mag": mixing_params["bmix_mag"],

        # DFT+U parameters
        "ldau": ldau,
        "ldautype": ldautype,
        "ldau_luj": ldau_luj,

        # Geometry optimization parameters
        "ibrion": opt_params["ibrion"],
        "potim": opt_params["potim"],
        "nsw": opt_params["nsw"],

        # Setups
        "setups": vasp_params["setups"],
    }

    if calc_params["kspacing"] is not None:
        calc_params["kpts"] = type_params["kpts"]

    calc = Vasp(**calc_params)
    return calc

def set_lmaxmix(atoms=None):
    # lmaxmix setting
    symbols = atoms.get_chemical_symbols()
    d_elements = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
                  "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
                  "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
                  "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"]
    f_elements = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                  "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]

    atoms.calc.set(lmaxmix=2)  # default

    if len(set(symbols) & set(d_elements)) != 0:
        # has some d_elements
        atoms.calc.set(lmaxmix=4)

    if len(set(symbols) & set(f_elements)) != 0:
        # has some f_elements
        atoms.calc.set(lmaxmix=6)

    return None
