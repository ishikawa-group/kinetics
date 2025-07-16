import sys
import os
import argparse
import uuid
import logging
import json
from ase import Atom, Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.db import connect
from ase.data import atomic_numbers
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


def vegard_lattice_constant(elements, fractions=None):
    """
    elements : ['Pt','Ni', ...]
    fractions: [0.5,0.5] など。None の場合は等分
    """
    from ase.data import reference_states, atomic_numbers

    def _elemental_a(symbol: str) -> float:
        """ASE の reference_states から FCC 格子定数 (Å) を返す"""
        from ase.data import reference_states, atomic_numbers
        Z = atomic_numbers[symbol]
        a = reference_states[Z].get('a')  # FCC は 'a' キーに格子定数
        if a is None:
            raise ValueError(f"No reference lattice constant for {symbol}")
        return a

    n = len(elements)
    if fractions is None:
        fractions = [1.0 / n] * n
    if abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("Fractions must sum to 1")
    constants = [_elemental_a(el) for el in elements]
    return sum(a * x for a, x in zip(constants, fractions))


def make_bimetallic_alloys(
        num_samples=10,
        output_dir="./",
        size=[4, 4, 4],
        elements=["Pt", "Ni"],
        jsonfile="structures.json"
    ):

    # --- パラメータ設定 ---
    vacuum = None

    # --- 出力先ディレクトリの設定 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 出力ファイルのパス（jsonを指定されたフォルダ内に出力）
    db_path = os.path.join(output_dir, jsonfile)
    if os.path.exists(db_path):
        os.remove(db_path)

    db = connect(db_path)

    # ランダムシード設定（再現性のため）
    np.random.seed(42)

    # generation just to have the number of atoms in the slab
    num_atoms = len(fcc111(symbol=elements[0], size=size, a=2.5, vacuum=vacuum, periodic=True))

    print(f"Generating {num_samples} random alloy structures...")

    # --- 指定された数だけ構造生成 ---
    for i in range(num_samples):
        # fraction of main and sub elements
        main_fraction = np.random.uniform(1/num_atoms, (num_atoms - 1)/num_atoms)
        sub_fraction = 1 - main_fraction
        fractions = [main_fraction, sub_fraction]

        # Vegard法で格子定数を計算
        lattice_const = vegard_lattice_constant(elements, fractions)

        # fcc111Bulkの作成（基本はmainの構造）
        surf = fcc111(symbol=elements[0], size=size, a=lattice_const, vacuum=vacuum, periodic=True)

        # --- 合金組成のランダム配置 ---
        num_main = int(round(num_atoms * main_fraction))
        num_sub = int(round(num_atoms * sub_fraction))

        # 原子番号リストを作成
        alloy_list = [atomic_numbers[elements[0]]] * num_main + [atomic_numbers[elements[1]]] * num_sub

        # ランダムにシャッフルして各元素をランダム配置
        np.random.shuffle(alloy_list)
        surf.set_atomic_numbers(alloy_list)

        # --- EMT計算器の設定（FutureWarning解消のため calc 属性を直接代入） ---
        surf.calc = EMT()

        # --- 表面情報の取得 ---
        ads_info = surf.info["adsorbate_info"]

        # --- データベースへの書き込み ---
        data = {
            "chemical_formula": surf.get_chemical_formula(),
            "main_fraction": float(main_fraction),
            "sub_fraction": float(sub_fraction),
            "lattice_constant": float(lattice_const),
            "run": i,
            "adsorbate_info": ads_info
        }

        db.write(surf, data=data)

        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/{num_samples} structures")

    print(f"Structures saved to {db_path}")


def read_reactionfile(file):
    """
    Read reaction file and return reactants, reactions, and products.

    Args:
        file (str): reaction file name
    Returns:
        reac (list): reactants
        rxn (list): reaction type
        prod (list): products
    """
    import re

    f = open(file, "r")

    # drop comment and branck lines
    lines = f.readlines()
    newlines = []
    for line in lines:
        if not (re.match(r"^#", line)) and not (re.match(r"^s*$", line)):
            newlines.append(line)
    lines = newlines
    numlines = len(lines)

    reac = list(range(numlines))
    rxn = list(range(numlines))
    prod = list(range(numlines))

    for i, line in enumerate(lines):
        text = line.replace("\n", "").replace(">", "").split("--")
        reac_tmp = text[0]
        rxn_tmp = text[1]
        prod_tmp = text[2]

        reac[i] = re.split(" \+ ", reac_tmp)  # for cations
        prod[i] = re.split(" \+ ", prod_tmp)  # for cations

        reac[i] = remove_space(reac[i])
        prod[i] = remove_space(prod[i])

        rxn[i] = reac[i][0] + "_" + rxn_tmp

    return reac, rxn, prod


def return_lines_of_reactionfile(file):
    """
    Return lines of reaction file.
    """
    import re

    # drop comment and branck lines
    f = open(file, "r")
    lines = f.readlines()
    newlines = []
    for line in lines:
        if not (re.match(r"^#", line)) and not (re.match(r"^s*$", line)):
            newlines.append(line)
    lines = newlines
    return lines


def remove_space(obj):
    """
    Remove space in the object.
    """
    newobj = [0] * len(obj)
    if isinstance(obj, str):
        #
        # string
        #
        newobj = obj.replace(" ", "")
    elif isinstance(obj, list):
        #
        # list
        #
        for i, obj2 in enumerate(obj):
            if isinstance(obj2, list):
                #
                # nested list
                #
                for ii, jj in enumerate(obj2):
                    jj = jj.strip()
                newobj[i] = jj
            elif isinstance(obj2, str):
                #
                # simple list
                #
                obj2 = obj2.replace(" ", "")
                newobj[i] = obj2
            elif isinstance(obj2, int):
                #
                # integer
                #
                newobj[i] = obj2
            else:
                newobj[i] = obj2
    else:  # error
        print("remove_space: input str or list")

    return newobj


def get_reac_and_prod(reactionfile):
    """
    Form reactant and product information.
    """
    (reac, rxn, prod) = read_reactionfile(reactionfile)

    rxn_num = len(rxn)

    r_ads = list(range(rxn_num))
    r_site = [[] for _ in range(rxn_num)]
    r_coef = [[] for _ in range(rxn_num)]

    p_ads = list(range(rxn_num))
    p_site = list(range(rxn_num))
    p_coef = list(range(rxn_num))

    for irxn, jrnx in enumerate(rxn):
        ireac = reac[irxn]
        iprod = prod[irxn]
        ireac_num = len(ireac)
        iprod_num = len(iprod)
        #
        # reactant
        #
        r_ads[irxn] = list(range(ireac_num))
        r_site[irxn] = list(range(ireac_num))
        r_coef[irxn] = list(range(ireac_num))

        for imol, mol in enumerate(ireac):
            r_site[irxn][imol] = []
            r_ads[irxn][imol] = []
            #
            # coefficient
            #
            if "*" in mol:
                # r_coef[irxn][imol] = int(mol.split("*")[0])
                r_coef[irxn][imol] = float(mol.split("*")[0])
                rest = mol.split("*")[1]
            else:
                r_coef[irxn][imol] = 1
                rest = mol

            # site
            if ',' in rest:
                sites = rest.split(',')
                for isite, site in enumerate(sites):
                    r_site[irxn][imol].append(site.split('_')[1])
                    r_ads[irxn][imol].append(site.split('_')[0])
            elif '_' in rest:
                r_site[irxn][imol].append(rest.split('_')[1])
                r_ads[irxn][imol].append(rest.split('_')[0])
            else:
                r_site[irxn][imol].append('gas')
                r_ads[irxn][imol].append(rest)
        #
        # product
        #
        p_ads[irxn] = list(range(iprod_num))
        p_site[irxn] = list(range(iprod_num))
        p_coef[irxn] = list(range(iprod_num))

        for imol, mol in enumerate(iprod):
            p_site[irxn][imol] = []
            p_ads[irxn][imol] = []
            #
            # coefficient
            #
            if "*" in mol:
                # p_coef[irxn][imol] = int(mol.split("*")[0])
                p_coef[irxn][imol] = float(mol.split("*")[0])
                rest = mol.split("*")[1]
            else:
                p_coef[irxn][imol] = 1
                rest = mol

            # site
            if ',' in rest:
                sites = rest.split(',')
                for isite, site in enumerate(sites):
                    p_site[irxn][imol].append(site.split('_')[1])
                    p_ads[irxn][imol].append(site.split('_')[0])
            elif '_' in rest:
                p_site[irxn][imol].append(rest.split('_')[1])
                p_ads[irxn][imol].append(rest.split('_')[0])
            else:
                p_site[irxn][imol].append('gas')
                p_ads[irxn][imol].append(rest)

    return r_ads, r_site, r_coef, p_ads, p_site, p_coef


def get_number_of_reaction(reactionfile):
    """
    Return number of elementary reactions
    """
    (reac, rxn, prod) = read_reactionfile(reactionfile)
    rxn_num = len(rxn)
    return rxn_num


def get_preexponential(reactionfile):
    """
    Return pre-exponential factor.
    Not needed for MATLAB use
    """
    from ase.collections import methane

    #
    # calculate pre-exponential factor
    #
    (r_ads, r_site, r_coef, p_ads, p_site, p_coef) = get_reac_and_prod(reactionfile)

    rxn_num = get_number_of_reaction(reactionfile)

    Afor = np.array(rxn_num, dtype="f")
    Arev = np.array(rxn_num, dtype="f")

    mass_reac = np.array(rxn_num * [range(len(r_ads[0]))], dtype="f")
    mass_prod = np.array(rxn_num * [range(len(p_ads[0]))], dtype="f")

    for irxn in range(rxn_num):
        #
        # reactants
        #
        for imol, mol in enumerate(r_ads[irxn]):
            tmp = methane[mol]
            mass = sum(tmp.get_masses())
            mass_reac[irxn, imol] = mass
        #
        # products
        #
        for imol, mol in enumerate(p_ads[irxn]):
            tmp = methane[mol]
            mass = sum(tmp.get_masses())
            mass_prod[irxn, imol] = mass

        Afor[irxn] = 1.0
        Arev[irxn] = 1.0

    return Afor, Arev


def get_rate_coefficient(reactionfile, Afor, Arev, Efor, Erev, T):
    """
    Return rate coefficient.
    Not needed for MATLAB use
    """
    # calculate rate constant
    # (r_ads, r_site, r_coef,  p_ads, p_site, p_coef) = get_reac_and_prod(reactionfile)

    rxn_num = get_number_of_reaction(reactionfile)

    kfor = np.array(rxn_num, dtype="f")
    krev = np.array(rxn_num, dtype="f")

    RT = 8.314 * T / 1000.0  # in kJ/mol

    for irxn in range(rxn_num):
        kfor[irxn] = Afor[irxn] * np.exp(-Efor[irxn] / RT)
        krev[irxn] = Arev[irxn] * np.exp(-Erev[irxn] / RT)

    return kfor, krev


def read_speciesfile(speciesfile):
    """
    read species
    """
    f = open(speciesfile)

    species = f.read()
    species = species.replace('[', '')
    species = species.replace(']', '')
    species = species.replace(' ', '')
    species = species.replace('\n', '')
    species = species.replace('\'', '')
    species = species.split(",")

    return species


def remove_parentheses(file):
    """
    remove parentheses -- maybe for MATLAB use
    """
    import os
    tmpfile = "ttt.txt"
    os.system('cat %s | sed "s/\[//g" > %s' % (file, tmpfile))
    os.system('cat %s | sed "s/\]//g" > %s' % (tmpfile, file))
    os.system('rm %s' % tmpfile)


def get_species_num(*species):
    """
    Return what is the number of species in speciesfile.
    If argument is not present, returns the number of species.
    """
    from reaction_tools import read_speciesfile

    speciesfile = "species.txt"
    lst = read_speciesfile(speciesfile)

    if len(species) == 0:
        # null argument: number of species
        return len(lst)
    else:
        # return species number
        spec = species[0]
        return lst.index(spec)


def get_adsorption_sites(infile):
    """
    Read adsorption sites.
    """
    from reaction_tools import remove_space

    f = open(infile, "r")

    lines = f.readlines()

    mol = list(range(len(lines)))
    site = list(range(len(lines)))

    for i, line in enumerate(lines):
        aaa, bbb = line.replace("\n", "").split(":")
        mol[i] = remove_space(aaa)
        bbb = remove_space(bbb)
        site[i] = bbb.split(",")

    return mol, site


def find_closest_atom(surf, offset=(0, 0)):
    """
    Find the closest atom to the adsorbate.
    """
    from ase.build import add_adsorbate

    dummy = Atom('H', (0, 0, 0))
    ads_height = 0.1
    add_adsorbate(surf, dummy, ads_height, position=(0, 0), offset=offset)
    natoms = len(surf.get_atomic_numbers())
    last = natoms - 1
    # ads_pos = surf.get_positions(last)

    dist = surf.get_distances(last, [range(natoms)], vector=False)
    dist = np.array(dist)
    dist = np.delete(dist, last)  # delete adsorbate itself
    clothest = np.argmin(dist)

    return clothest


def sort_atoms_by_z(atoms, elementwise=True):
    """
    Sort atoms by z-coordinate.

    Args:
        atoms: ASE Atoms object.
        elementwise: Whether to sort by element-wise gruop. True or False.
    Returns:
        newatoms: Sorted Atoms object.
        zcount: Element-wise list of each atoms index.
    """
    import collections

    #
    # keep information for original Atoms
    #
    tags = atoms.get_tags()
    pbc = atoms.get_pbc()
    cell = atoms.get_cell()
    natoms = len(atoms)

    dtype = [("idx", int), ("z", float)]
    #
    # get set of chemical symbols
    #
    symbols = atoms.get_chemical_symbols()
    elements = sorted(set(symbols), key=symbols.index)
    num_elem = []
    for i in elements:
        num_elem.append(symbols.count(i))

    #
    # loop over each groups
    #
    iatm = 0
    newatoms = Atoms()
    zcount = []

    if elementwise:
        for inum in num_elem:
            zlist = np.array([], dtype=dtype)
            for idx in range(inum):
                tmp = np.array([(iatm, atoms[iatm].z)], dtype=dtype)
                zlist = np.append(zlist, tmp)
                iatm += 1

            zlist = np.sort(zlist, order="z")

            for i in zlist:
                idx = i[0]
                newatoms.append(atoms[idx])

            tmp = np.array([], dtype=float)
            for val in zlist:
                tmp = np.append(tmp, round(val[1], 2))
            tmp = collections.Counter(tmp)
            zcount.append(list(tmp.values()))

    else:
        zlist = np.array([], dtype=dtype)
        for idx in range(natoms):
            tmp = np.array([(iatm, atoms[iatm].z)], dtype=dtype)
            zlist = np.append(zlist, tmp)
            iatm += 1

        zlist = np.sort(zlist, order="z")

        for i in zlist:
            idx = i[0]
            newatoms.append(atoms[idx])

    #
    # restore tag, pbc, cell
    #
    newatoms.set_tags(tags)
    newatoms.set_pbc(pbc)
    newatoms.set_cell(cell)

    return newatoms, zcount


def get_number_of_valence_electrons(atoms):
    """
    Returns number of valence electrons for VASP calculation.
    Calls VASP calculation once.
    Returned electron numbers should be ++1 or --1 for cations, anions, etc.
    """
    from ase.calculators.vasp import Vasp
    atoms.calc = Vasp(prec="normal", xc="PBE", encut=300.0, nsw=0, nelm=1)
    nelec = atoms.calc.get_number_of_electrons()
    return nelec


def read_charge(mol):
    """
    Read charge from molecule.
    """
    charge = 0.0  # initial
    if "^" in mol:
        neutral = False
        mol, charge = mol.split('^')
        charge = float(charge.replace('{', '').replace('}', '').replace('+', ''))
    else:
        neutral = True

    return mol, neutral, charge


def remove_side_and_flip(mol):
    """
    Remove SIDE and FLIP in molecule
    """
    if '-SIDEx' in mol:
        mol = mol.replace('-SIDEx', '')
    elif '-SIDEy' in mol:
        mol = mol.replace('-SIDEy', '')
    elif '-SIDE' in mol:
        mol = mol.replace('-SIDE', '')
    elif '-FLIP' in mol:
        mol = mol.replace('-FLIP', '')
    elif '-TILT' in mol:
        mol = mol.replace('-TILT', '')
    elif '-HIGH' in mol:
        mol = mol.replace('-HIGH', '')

    return mol


def neb_copy_contcar_to_poscar(nimages):
    """
    Copy 0X/CONTCAR to 0X/POSCAR after NEB run.
    """
    import os
    for images in range(nimages):
        os.system('cp %02d/CONTCAR %02d/POSCAR' % (images + 1, images + 1))


def make_it_closer_by_exchange(atom1, atom2, thre=0.05):
    """
    Exchange atoms to make it closer.

    Args:
        atom1 (Atoms): Atoms object
        atom2 (Atoms): Atoms object
        thre: when distance is larger than this value, do switch
    """
    from ase.geometry import distance

    natoms = len(atom1)
    const_list = atom1.constraints[0].get_indices()
    # Constrained atoms. Do not exchange these.

    for i in range(natoms):
        for j in range(i + 1, natoms):
            if atom1[i].symbol == atom1[j].symbol:
                # measure distance between "ATOMS" (in ASE object)
                dis_bef = distance(atom1, atom2)
                atom1B = atom1.copy()
                atom1B.positions[[i, j]] = atom1B.positions[[j, i]]
                dis_aft = distance(atom1B, atom2)

                if (dis_aft + thre < dis_bef) and not (i in const_list) and not (j in const_list):
                    #
                    # switch
                    #
                    print("exchanged {0:3d} and {1:3d}: (dis_bef, dis_aft) = ({2:6.2f},{3:6.2f})".
                          format(i, j, dis_bef, dis_aft))
                    tmp = atom1B.numbers[i]
                    atom1B.numbers[i] = atom1B.numbers[j]
                    atom1B.numbers[j] = tmp

                    atom1 = atom1B.copy()

    return atom1


def get_adsorbate_type(adsorbate, site):
    """
    Returns adsorbate type.

    Args:
        adsorbate (str): Molecule name.
        site (str): Site name.
    Returns:
        adsorbate_type (str): Adsorbate type
            "gaseous": gaseous molecule
            "surface": bare surface
            "adsorbed": adsorbed molecule
    """
    if site == "gas":
        if "surf" in adsorbate:
            ads_type = "surface"
        else:
            ads_type = "gaseous"
    else:
        ads_type = "adsorbed"

    return ads_type


def make_surface_from_cif(
        cif_file: str,
        layers: int=2,
        indices: list=[0, 0, 1],
        repeat: list=[1, 1, 1],
        vacuum: float=8.0) -> Atoms:
    """
    Make a surface from a CIF file.
    """
    from ase.build import surface
    from ase.io import read
    from ase.visualize import view

    bulk = read(cif_file)
    bulk = bulk*repeat
    surf = surface(bulk, indices=indices, layers=layers, vacuum=vacuum)

    surf.translate([0, 0, -vacuum+0.1])
    surf.pbc = True

    return surf


def replace_element(atoms, from_element: str, to_element: str, percent=100):
    import random

    from ase.build import sort

    elements = atoms.get_chemical_symbols()
    num_from_elements = elements.count(from_element)
    num_replace = int((percent/100) * num_from_elements)

    indices = [i for i, j in enumerate(elements) if j == from_element]
    random_item = random.sample(indices, num_replace)
    for i in random_item:
        atoms[i].symbol = to_element

    atoms = sort(atoms)
    return atoms


def run_packmol(xyz_file, a, num, outfile):
    import os

    packmol = "/Users/ishi/packmol/packmol"
    filetype = "xyz"

    cell1 = [0.0, 0.0, 0.0, a, a, a]
    cell2 = " ".join(map(str, cell1))

    f = open("pack_tmp.inp", "w")
    text = ["tolerance 2.0"             + "\n",
            "output "     + outfile     + "\n",
            "filetype "   + filetype    + "\n",
            "structure "  + xyz_file    + "\n",
            "  number "   + str(num)    + "\n",
            "  inside box " + cell2     + "\n",
            "end structure"]
    f.writelines(text)
    f.close()

    run_string = packmol + " < pack_tmp.inp"

    os.system(run_string)

    # os.system("rm pack_tmp.inp")


def json_to_csv(jsonfile, csvfile):
    import json

    import pandas as pd
    from pandas.io.json import json_normalize
    f = open(jsonfile, "r")
    d = json.load(f)

    dd = []
    nrec = len(d)
    for i in range(1, nrec):
        if str(i) in d:
            tmp = d[str(i)]
            dd.append(json_normalize(tmp))

    ddd = pd.concat(dd)

    newcol = []
    for key in ddd.columns:
        key = key.replace("calculator_parameters.", "")
        key = key.replace("key_value_pairs.", "")
        key = key.replace("data.", "")
        newcol.append(key)

    ddd.columns = newcol

    # sort data by "num"
    if "num" in ddd.columns:
        ddd2 = ddd.set_index("num")
        ddd  = ddd2.sort_index()

    ddd.to_csv(csvfile)


def load_ase_json(jsonfile):
    import json

    import pandas as pd
    f = open(jsonfile, "r")
    d = json.load(f)

    dd = []
    nrec = len(d)
    for i in range(1, nrec):
        if str(i) in d:
            tmp = d[str(i)]
            dd.append(pd.json_normalize(tmp))

    ddd = pd.concat(dd)

    newcol = []
    for key in ddd.columns:
        key = key.replace("calculator_parameters.", "")
        key = key.replace("key_value_pairs.", "")
        key = key.replace("data.", "")
        newcol.append(key)

    ddd.columns = newcol

    # sort data by "num"
    if "num" in ddd.columns:
        ddd2 = ddd.set_index("num")
        ddd  = ddd2.sort_index()

    return ddd


def delete_num_from_json(num, jsonfile):
    from ase.db import connect

    db = connect(jsonfile)
    id_ = db.get(num=num).id
    db.delete([id_])


def sort_atoms_by(atoms, xyz="x", elementwise=True):
    # keep information for original Atoms
    tags = atoms.get_tags()
    pbc  = atoms.get_pbc()
    cell = atoms.get_cell()
    dtype = [("idx", int), (xyz, float)]

    newatoms = Atoms()
    symbols = list(set(atoms.get_chemical_symbols()))
    if elementwise:
        for symbol in symbols:
            subatoms = Atoms(list(filter(lambda x: x.symbol == symbol, atoms)))
            atomlist = np.array([], dtype=dtype)
            for idx, atom in enumerate(subatoms):
                if xyz == "x":
                    tmp = np.array([(idx, atom.x)], dtype=dtype)
                elif xyz == "y":
                    tmp = np.array([(idx, atom.y)], dtype=dtype)
                else:
                    tmp = np.array([(idx, atom.z)], dtype=dtype)

                atomlist = np.append(atomlist, tmp)

            atomlist = np.sort(atomlist, order=xyz)

            for i in atomlist:
                idx = i[0]
                newatoms.append(subatoms[idx])
    else:
        atomlist = np.array([], dtype=dtype)
        for idx, atom in enumerate(atoms):
            if xyz == "x":
                tmp = np.array([(idx, atom.x)], dtype=dtype)
            elif xyz == "y":
                tmp = np.array([(idx, atom.y)], dtype=dtype)
            else:
                tmp = np.array([(idx, atom.z)], dtype=dtype)

            atomlist = np.append(atomlist, tmp)

        atomlist = np.sort(atomlist, order=xyz)

        for i in atomlist:
            idx = i[0]
            newatoms.append(atoms[idx])

    # restore
    newatoms.set_tags(tags)
    newatoms.set_pbc(pbc)
    newatoms.set_cell(cell)

    return newatoms


def get_number_of_layers(atoms):
    symbols = list(set(atoms.get_chemical_symbols()))
    symbols = sorted(symbols)
    nlayers = []

    for symbol in symbols:
        subatoms = Atoms(list(filter(lambda x: x.symbol == symbol, atoms)))
        pos  = subatoms.positions
        zpos = np.round(pos[:, 2], decimals=4)
        nlayers.append(len(list(set(zpos))))

    return nlayers


def set_tags_by_z(atoms, elementwise=True):
    import pandas as pd

    pbc  = atoms.get_pbc()
    cell = atoms.get_cell()

    newatoms = Atoms()
    symbols = list(set(atoms.get_chemical_symbols()))
    symbols = sorted(symbols)

    if elementwise:
        for symbol in symbols:
            subatoms = Atoms(list(filter(lambda x: x.symbol == symbol, atoms)))
            pos  = subatoms.positions
            zpos = np.round(pos[:, 2], decimals=1)
            bins = list(set(zpos))
            bins = np.sort(bins)
            bins = np.array(bins) + 1.0e-2
            bins = np.insert(bins, 0, 0)

            labels = []
            for i in range(len(bins)-1):
                labels.append(i)

            tags = pd.cut(zpos, bins=bins, labels=labels).tolist()

            subatoms.set_tags(tags)
            newatoms += subatoms
    else:
        subatoms = atoms.copy()
        pos  = subatoms.positions
        zpos = np.round(pos[:, 2], decimals=1)
        bins = list(set(zpos))
        bins = np.sort(bins)
        bins = np.array(bins) + 1.0e-2
        bins = np.insert(bins, 0, 0)

        labels = []
        for i in range(len(bins)-1):
            labels.append(i)

        tags = pd.cut(zpos, bins=bins, labels=labels).tolist()

        subatoms.set_tags(tags)
        newatoms += subatoms

    # restore
    newatoms.set_pbc(pbc)
    newatoms.set_cell(cell)

    return newatoms


def remove_layers(atoms=None, element=None, layers_to_remove=1):
    """
    Remove layers of symbol at high-in-z.

    Args:
        atoms (Atoms): Atoms object
        element (str): Element symbol. If None, any atom can be deleted.
        layers_to_remove(int): Number of layers to remove, from top of the surface.
    """
    pbc  = atoms.get_pbc()
    cell = atoms.get_cell()

    atoms_copy = atoms.copy()
    atoms_copy = sort_atoms_by(atoms_copy, xyz="z")

    if element is not None:
        atoms_copy = set_tags_by_z(atoms_copy)
    else:
        atoms_copy = set_tags_by_z(atoms_copy, elementwise=False)

    newatoms = Atoms()

    tags = atoms_copy.get_tags()
    if element is not None:
        cond = [i == element for i in atoms_copy.get_chemical_symbols()]
        maxtag = max(list(tags[cond]))
    else:
        maxtag = max(atoms_copy.get_tags())

    for i, atom in enumerate(atoms_copy):
        if element is not None:
            if atom.tag >= maxtag - layers_to_remove + 1 and atom.symbol == element:
                # remove this atom
                pass
            else:
                newatoms += atom
        else:
            if atom.tag >= maxtag - layers_to_remove + 1:
                # remove this atom
                pass
            else:
                newatoms += atom

    newatoms.set_pbc(pbc)
    newatoms.set_cell(cell)

    return newatoms


def fix_lower_surface(atoms, adjust_layer=None):
    """
    Fix lower surface atoms. By default, lower half (controled by tag) is fixed.

    Args:
        atoms (Atoms): Atoms object
        adjust_layer (list): List of element-wise layers to fix. Positive means more layers are fixed.
    """
    from ase.constraints import FixAtoms

    newatoms = atoms.copy()
    newatoms = sort_atoms_by(newatoms, xyz="z")  # sort
    newatoms = set_tags_by_z(newatoms)  # set tags

    # prepare symbol dict
    symbols_ = list(set(atoms.get_chemical_symbols()))
    symbols_ = sorted(symbols_)
    symbols = {}
    for i, sym in enumerate(symbols_):
        symbols.update({sym: i})

    # Determine fixlayer, which is a list of elements. Half of nlayers.
    nlayers = get_number_of_layers(newatoms)

    # check
    div = [i // 2 for i in nlayers]
    mod = [i % 2 for i in nlayers]

    fixlayers = [i + j for (i, j) in zip(div, mod)]

    if adjust_layer is not None:
        fixlayers = [sum(x) for x in zip(fixlayers, adjust_layer)]

    fixlist = []  # list of fixed atoms

    # tags = newatoms.get_tags()
    # minind = np.argmin(tags)
    # maxind = np.argmax(tags)

    # lowest_z  = newatoms[minind].position[2]
    # highest_z = newatoms[maxind].position[2]
    # z_thre = (highest_z - lowest_z) / 2 + lowest_z

    for iatom in newatoms:
        ind = symbols[iatom.symbol]
        # z_pos = iatom.position[2]

        # if iatom.tag < fixlayers[ind] and z_pos < z_thre:
        if iatom.tag < fixlayers[ind]:
            fixlist.append(iatom.index)
        else:
            pass

    constraint = FixAtoms(indices=fixlist)
    newatoms.constraints = constraint

    return newatoms


def find_highest(json, score):
    import pandas as pd

    df = pd.read_json(json)
    df = df.set_index("unique_id")
    df = df.dropna(subset=[score])
    df = df.sort_values(score, ascending=False)

    best = df.iloc[0].name

    return best


def make_step(atoms):
    newatoms = atoms.copy()
    newatoms = sort_atoms_by(newatoms, xyz="z")

    nlayer = get_number_of_layers(newatoms)
    nlayer = nlayer[0]
    perlayer  = len(newatoms) // nlayer
    toplayer  = newatoms[-perlayer:]
    top_layer = sort_atoms_by(toplayer, xyz="y")

    # first remove top layer then add sorted top layer
    del newatoms[-perlayer:]
    newatoms += top_layer

    remove = perlayer // 2

    nstart = perlayer*(nlayer-1)  # index for the atom starting the top layer
    del newatoms[nstart:nstart+remove]

    return newatoms


def mirror_invert(atoms, direction="x"):
    """
    Mirror invert the surface in the specified direction.

    Args:
        atoms: Atoms object
        direction: "x", "y", or "z"
    """
    pos  = atoms.get_positions()
    cell = atoms.cell

    # set position and cell
    if direction == "x":
        pos[:, 0] = -pos[:, 0]
        cell = [[-cell[i][0], cell[i][1], cell[i][2]] for i in range(3)]
    elif direction == "y":
        pos[:, 1] = -pos[:, 1]
        cell = [[cell[i][0], -cell[i][1], cell[i][2]] for i in range(3)]
    elif direction == "z":
        highest_z = pos[:, 2].max()
        atoms.translate([0, 0, -highest_z])
        pos[:, 2] = -pos[:, 2]
        cell = [[cell[i][0], cell[i][1], -cell[i][2]] for i in range(3)]
    else:
        print("direction must be x, y, or z")
        quit()

    atoms.set_positions(pos)

    cell = np.array(cell)
    cell = np.round(cell + 1.0e-5, decimals=4)
    atoms.set_cell(cell)

    return atoms


def make_barplot(labels=None, values=None, threshold=100, ylabel="y-value",
                 fontsize=16, filename="bar_plot.png"):
    """
    Make a bar plot of values with labels, filtering out values above the threshold.

    Args:
        labels: List of labels for the x-axis
        values: List of values for the y-axis
        threshold: Maximum value to include in the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [label for label, value in zip(labels, values) if value < threshold]
    values = [value for value in values if value < threshold]

    sorted_indices = np.argsort(values)
    sorted_labels  = [labels[i] for i in sorted_indices]
    sorted_values  = [values[i] for i in sorted_indices]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)

    plt.bar(sorted_labels, sorted_values, color="skyblue")

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)

    ax.xaxis.set_tick_params(direction="out", labelsize=fontsize, width=2, pad=10)
    ax.yaxis.set_tick_params(direction="out", labelsize=fontsize, width=2, pad=10)
    ax.set_ylabel(ylabel, fontsize=fontsize+4, labelpad=20)

    plt.xticks(rotation=45, verticalalignment="top", horizontalalignment="right")
    plt.savefig(filename, dpi=300, bbox_inches="tight")


def make_energy_diagram(deltaEs=None, has_barrier=False, rds=1, savefig=True,
                        figname="ped.png", xticklabels=None):
    """Generate potential energy diagram for reaction steps.

    Args:
        deltaEs (list or numpy.ndarray): Energy differences between reaction steps
        has_barrier (bool, optional): Include transition state barrier. Defaults to False.
        rds (int, optional): Index of rate determining step for barrier calculation. Defaults to 1.
        savefig (bool, optional): Save the figure to a file. Defaults to True.
        figname (str, optional): Output filename. Defaults to "ped.png".
        xticklabels (list, optional): Labels for reaction steps. Defaults to None.

    Returns:
        tuple: Plot data (x coordinates array, y coordinates array)

    Note:
        When has_barrier is True, the function calculates transition state barrier
        using Brønsted-Evans-Polanyi (BEP) relationship:
        Ea = alpha * deltaE + beta
        where alpha = 0.87 and beta = 1.34
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import interpolate

    # Convert input to numpy array
    deltaEs = np.array(deltaEs)
    num_rxn = len(deltaEs)

    # Calculate cumulative energies
    ped = np.zeros(1)
    for i in range(num_rxn):
        ped = np.append(ped, ped[-1] + deltaEs[i])

    y1 = ped

    # Handle transition state barriers using BEP relationship
    if has_barrier:
        alpha = 0.87  # BEP parameters
        beta = 1.34
        Ea = y1[rds] * alpha + beta  # Calculate barrier height

        # Extend x length after transition state curve
        y1 = np.insert(y1, rds, y1[rds])
        num_rxn += 1

    # Generate interpolation coordinates
    points = 500  # Number of points for smooth curve
    x1_latent = np.linspace(-0.5, num_rxn + 0.5, points)
    x1 = np.arange(0, num_rxn + 1)
    f1 = interpolate.interp1d(x1, y1, kind="nearest", fill_value="extrapolate")

    # Replace rate determining step by quadratic curve for barrier
    if has_barrier:
        x2 = [rds - 0.5, rds, rds + 0.5]
        x2 = np.array(x2)
        y2 = np.array([y1[rds-1], Ea, y1[rds+1]])
        f2 = interpolate.interp1d(x2, y2, kind="quadratic")

    # Combine nearest neighbor interpolation with barrier curve
    y = np.array([])
    for i in x1_latent:
        val1 = f1(i)
        val2 = -1.0e10
        if has_barrier:
            val2 = f2(i)
        y = np.append(y, max(val1, val2))

    # Generate and save the plot
    if savefig:
        # Set plot style
        sns.set(style="darkgrid", rc={"lines.linewidth": 2.0, "figure.figsize": (10, 4)})

        # Create plot
        p = sns.lineplot(x=x1_latent, y=y, sizes=(0.5, 1.0))

        # Set labels and font sizes
        p.set_xlabel("Steps", fontsize=16)
        p.set_ylabel("Energy (eV)", fontsize=16)
        p.tick_params(axis="both", labelsize=14)
        p.yaxis.set_major_formatter(lambda x, p: f"{x:.1f}")

        # Set x-axis labels if provided
        if xticklabels is not None:
            xticklabels.insert(0, "dummy")
            p.set_xticklabels(xticklabels, rotation=45, ha="right")

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(figname)

    return x1_latent, y


def add_data_to_jsonfile(jsonfile, data):
    """
    add data to database
    """
    import json
    import os

    if not os.path.exists(jsonfile):
        with open(jsonfile, "w") as f:
            json.dump([], f)

    with open(jsonfile, "r") as f:
        datum = json.load(f)

        # remove "doing" record as calculation is done
        for i in range(len(datum)):
            if datum[i]["status"] == "doing":
                datum.pop(i)
                break

        datum.append(data)

    with open(jsonfile, "w") as f:
        json.dump(datum, f, indent=4)


def get_row_by_unique_id(db, unique_id: str) -> Any:
    """Find row by unique_id using multiple search methods.

    Args:
        db: ASE database connection
        unique_id: Unique identifier for the structure

    Returns:
        Database row matching the unique_id
    """
    logger = logging.getLogger(__name__)

    search_methods = [
        (lambda: list(db.select(f'unique_id="{unique_id}"')), "string"),
        (lambda: list(db.select(unique_id=unique_id)), "unique_id"),
        (lambda: [r for r in db.select() if r.key_value_pairs.get('unique_id') == unique_id], "key-value"),
        (lambda: [db.get(int(unique_id))], "integer ID")
    ]

    for method, method_name in search_methods:
        try:
            rows = method()
            if rows:
                logger.info(f"Found unique_id '{unique_id}' using {method_name}")
                return rows[0]
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Search method '{method_name}' failed: {e}")
            continue

    return None


def load_results_from_json(out_json: str) -> Tuple[List[Dict], List[str]]:
    """Load existing results from JSON file.

    Args:
        out_json: Path to JSON output file

    Returns:
        Tuple of (results list, unique_ids list)
    """
    logger = logging.getLogger(__name__)

    json_path = Path(out_json)
    if not json_path.exists() or json_path.stat().st_size == 0:
        logger.info(f"No existing results file found at {out_json}")
        return [], []

    try:
        with open(out_json) as f:
            existing = json.load(f)
        results = existing if isinstance(existing, list) else [existing]
        ids = [entry.get('unique_id') for entry in results]
        logger.info(f"Loaded {len(results)} existing results from {out_json}")
        return results, ids
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading results from {out_json}: {e}")
        return [], []


def save_results_to_json(out_json: str, results: List[Dict]) -> None:
    """Save results to JSON file.

    Args:
        out_json: Path to JSON output file
        results: List of result dictionaries to save
    """
    logger = logging.getLogger(__name__)

    try:
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved {len(results)} results to {out_json}")
    except IOError as e:
        logger.error(f"Error saving results to {out_json}: {e}")
        raise

    return None


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def set_initial_magmoms(atoms: Atoms):
    """Set initial magnetic moments based on element type."""
    magmom_dict = {
        "Fe": 5.0, "Co": 3.0, "Ni": 2.0, "Mn": 5.0, "Cr": 4.0,
        "O": 0.0, "H": 0.0, "C": 0.0, "N": 0.0, "S": 0.0
    }
    magmoms = []
    for symbol in atoms.get_chemical_symbols():
        magmoms.append(magmom_dict.get(symbol, 0.0))

    atoms.set_initial_magnetic_moments(magmoms)
    return None


def my_calculator(
        atoms,
        kind: str,
        calculator: str = "mace",
        yaml_path: str = "data/vasp.yaml",
        calc_directory: str = "calc"
):
    """
    Create calculator instance based on parameters from YAML file and attach to atoms.

    Args:
        atoms: ASE atoms object
        kind: "gas" / "slab" / "bulk"
        calculator: "vasp" / "mattersim" / "mace"- calculator type
        yaml_path: Path to YAML configuration file
        calc_directory: Calculation directory for VASP

    Returns:
        atoms: Atoms object with calculator set (ExpCellFilter for bulk calculations)
    """
    import yaml
    import sys
    import torch

    calculator = calculator.lower()

    # optimizer options
    fmax = 0.10
    steps = 100

    if calculator == "vasp":
        from ase.calculators.vasp import Vasp

        # Load YAML file directly
        try:
            with open(yaml_path, 'r') as f:
                vasp_params = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: VASP parameter file not found at {yaml_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {yaml_path}: {e}")
            sys.exit(1)

        if kind not in vasp_params['kinds']:
            raise ValueError(f"Invalid kind '{kind}'. Must be one of {list(vasp_params['kinds'].keys())}")

        # Copy common parameters
        params = vasp_params['common'].copy()
        # Update with kind-specific parameters
        params.update(vasp_params['kinds'][kind])
        # Set function argument parameters
        params['directory'] = calc_directory

        # Convert kpts to tuple (ASE expects tuple)
        if 'kpts' in params and isinstance(params['kpts'], list):
            params['kpts'] = tuple(params['kpts'])

        # Set calculator to atoms object and return
        atoms.calc = Vasp(**params)
        # Automatically set lmaxmix
        atoms = auto_lmaxmix(atoms)

    elif calculator == "mattersim":
        from mattersim.forcefield.potential import MatterSimCalculator
        from ase.filters import ExpCellFilter
        from ase.optimize import FIRE

        device = "cuda" if torch.cuda.is_available() else "cpu"
        atoms.calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)

        # Apply CellFilter for bulk calculations
        if kind == "bulk":
            atoms = ExpCellFilter(atoms)

        # Perform structure optimization
        optimizer = FIRE(atoms)
        optimizer.run(fmax=fmax, steps=steps)

        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        else:
            atoms = atoms

    elif calculator == "mattersim-matpes-pbe-d3":
        # Import the custom function
        from mattersim_matpes import mattersim_matpes_d3_calculator
        from ase.filters import ExpCellFilter
        from ase.optimize import FIRE

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use custom calculator with D3 dispersion corrections
        calculator = mattersim_matpes_d3_calculator(
            device=device,
            dispersion=True,  # Enable D3 dispersion corrections
            damping="bj",
            dispersion_xc="pbe"
        )

        # 設定変更を防ぐプロキシクラスを実装
        class ProtectedCalculator:
            def __init__(self, calculator):
                self._calculator = calculator

            def __getattr__(self, name):
                if name == 'set':
                    def protected_set(*args, **kwargs):
                        print("Warning: Calculator settings are protected and cannot be modified")
                        return self  # 何も変更せずに自身を返す

                    return protected_set
                return getattr(self._calculator, name)

        # 保護されたカリキュレータをセット
        atoms.calc = ProtectedCalculator(calculator)

        # Apply CellFilter for bulk calculations
        if kind == "bulk":
            atoms = ExpCellFilter(atoms)

        # Perform structure optimization
        optimizer = FIRE(atoms)
        optimizer.run(fmax=fmax, steps=steps)

        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        else:
            atoms = atoms

    elif calculator == "mattersim-matpes-pbe":
        # Import the custom function
        from mattersim_matpes import mattersim_matpes_d3_calculator
        from ase.filters import ExpCellFilter
        from ase.optimize import FIRE

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use custom calculator with D3 dispersion corrections
        calculator = mattersim_matpes_d3_calculator(
            device=device,
            dispersion=False,  # Enable D3 dispersion corrections
        )

        # 設定変更を防ぐプロキシクラスを実装
        class ProtectedCalculator:
            def __init__(self, calculator):
                self._calculator = calculator

            def __getattr__(self, name):
                if name == 'set':
                    def protected_set(*args, **kwargs):
                        print("Warning: Calculator settings are protected and cannot be modified")
                        return self  # 何も変更せずに自身を返す

                    return protected_set
                return getattr(self._calculator, name)

        # 保護されたカリキュレータをセット
        atoms.calc = ProtectedCalculator(calculator)

        # Apply CellFilter for bulk calculations
        if kind == "bulk":
            atoms = ExpCellFilter(atoms)

        # Perform structure optimization
        optimizer = FIRE(atoms)
        optimizer.run(fmax=fmax, steps=steps)

        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        else:
            atoms = atoms

    elif calculator == "mace":
        from mace.calculators import mace_mp
        from ase.filters import ExpCellFilter
        from ase.optimize import FIRE

        device = "cuda" if torch.cuda.is_available() else "cpu"
        url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model"

        mace_calculator = mace_mp(model=url,
                                  dispersion=True,
                                  dispersion_xc="pbe",
                                  default_dtype="float64",
                                  device=device)

        # 設定変更を防ぐプロキシクラスを実装
        class ProtectedMaceCalculator:
            def __init__(self, calculator):
                self._calculator = calculator

            def __getattr__(self, name):
                if name == 'set':
                    def protected_set(*args, **kwargs):
                        return self  # 何も変更せずに自身を返す

                    return protected_set
                return getattr(self._calculator, name)

        # 保護されたカリキュレータをセット
        atoms.calc = ProtectedMaceCalculator(mace_calculator)

        # Apply CellFilter for bulk calculations
        if kind == "bulk":
            atoms = ExpCellFilter(atoms)

        # Perform structure optimization
        optimizer = FIRE(atoms)
        optimizer.run(fmax=fmax, steps=steps)

        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        else:
            atoms = atoms

    else:
        raise ValueError("calculator must be 'vasp' or 'mace'")

    return atoms


def auto_lmaxmix(atoms):
    """Automatically set lmaxmix when d/f elements are present"""
    d_elements = {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"
    }
    f_elements = {
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu", "Ac", "Th", "Pa", "U", "Np",
        "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"
    }

    symbols = set(atoms.get_chemical_symbols())

    if symbols & f_elements:
        lmaxmix_value = 6
    elif symbols & d_elements:
        lmaxmix_value = 4
    else:
        lmaxmix_value = 2

    atoms.calc.set(lmaxmix=lmaxmix_value)
    return atoms
