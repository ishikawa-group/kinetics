def set_vasp_calculator(atom_type="molecule", dfttype="gga", do_optimization=False):
    """
    Set up a calculator using VASP.

    Args:
        atom_type: "molecule", "surface" or "solid".
        do_optimization: True or False.
    """
    import json
    from ase.calculators.vasp import Vasp

    if atom_type == "molecule":
        kpts = [1, 1, 1]
        ismear = 0
        sigma = 0.05
        lreal = True
        ldipol = False
        idipol = None
    elif atom_type == "surface":
        kpt  = 1  # now using kspacing
        kpts = [kpt, kpt, 1]
        ismear = 1
        sigma = 0.1
        lreal  = True  # False will take very long time
        ldipol = True
        idipol = 3
    elif atom_type == "solid":
        kpt  = 1
        kpts = [kpt, kpt, kpt]
        ismear = 0
        sigma = 0.2
        lreal = False
        ldipol = False
        idipol = None
    else:
        print("some error")
        quit()

    # common setting
    xc = "pbe"
    encut = 450.0  # fails at 400-450?
    ediff  = 1.0e-4
    ediffg = -40.0e-2
    lorbit = 10
    algo = "Normal"
    # algo = "Fast"
    nelm = 30
    nelmin = 3
    npar = 2  # 2 # change according to the computational environment
    nsim = npar
    kpar = 1
    ispin = 2
    isym = 0  # switching off symmetry
    kgamma = True
    setups = {"Ca": "_sv", "K": "_sv", "Ba": "_sv", "Cr": "_sv", "Mn": "_sv", 
              "Fe": "_sv", "Cs": "_sv", "Rb": "_sv", "Sr": "_sv", "Er": "_3", "Y": "_sv",
              "Zr": "_sv", "Dy": "_3", "Sm": "_3", "Pa": "_s", "Tm": "_3", "Nd": "_3", "Ho": "_3",
              "Nb": "_sv"
             }
    lasph = False
    lwave = False
    lcharg = False

    amix = 0.4; amix_mag = 1.6; bmix = 1.0;     bmix_mag = 1.0     # default
    # amix = 0.2; amix_mag = 0.8; bmix = 1.0e-4;  bmix_mag = 1.0e-4  # linear mixing
    
    # DFT + U
    if "plus" in dfttype:
        ldau = True
        lasph = True
        ldautype = 2
        u_param_file = "data/u_parameter.json"
        with open(u_param_file) as f:
            ldau_luj = json.load(f)
    else:
        ldau = None
        ldautype = None
        ldau_luj = None

    # metagga
    if "meta" in dfttype:
        xc = "r2scan"
        lasph = True
        algo = "Normal"

    # geometry optimization related
    if do_optimization:
        ibrion = 2
        potim = 0.1
        nsw = 2
    else:
        ibrion = 0
        potim = 0.0
        nsw = 0

    calc = Vasp(prec="Normal", xc=xc, pp="pbe", encut=encut, ismear=ismear, sigma=sigma, ediff=ediff, ediffg=ediffg,
                ibrion=ibrion, potim=potim, nsw=nsw, algo=algo, ldipol=ldipol, idipol=idipol, setups=setups, lasph=lasph,
                ispin=ispin, npar=npar, nsim=nsim, nelmin=nelmin, nelm=nelm, lreal=lreal, lorbit=lorbit, kgamma=kgamma,
                ldau=ldau, ldautype=ldautype, ldau_luj=ldau_luj, isym=isym,
                lwave=lwave, lcharg=lcharg, kspacing=0.7, kpts=None, kpar=kpar,
                amix=amix, amix_mag=amix_mag, bmix=bmix, bmix_mag=bmix_mag,
                )

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
