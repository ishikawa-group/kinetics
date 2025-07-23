import sys
import os
import json
import logging
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from ase import Atom, Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.data import atomic_numbers, reference_states
from ase.db import connect

# Constants
DEFAULT_RANDOM_SEED = 42
DEFAULT_VACUUM = 10.0
DEFAULT_ADS_HEIGHT = 0.1
BEP_ALPHA = 0.87  # Brønsted-Evans-Polanyi relationship
BEP_BETA = 1.34
Z_PRECISION = 1  # decimal places for z-coordinate rounding

# Element sets for VASP LMAXMIX
D_ELEMENTS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"
}
F_ELEMENTS = {
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Ac", "Th", "Pa", "U", "Np",
    "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"
}
MAGMOM_DICT = {
    "Fe": 5.0, "Co": 3.0, "Ni": 2.0, "Mn": 5.0, "Cr": 4.0,
    "O": 0.0, "H": 0.0, "C": 0.0, "N": 0.0, "S": 0.0
}


def vegard_lattice_constant(elements, fractions=None):
    """
    elements : ['Pt','Ni', ...]
    fractions: [0.5,0.5] など。None の場合は等分
    """
    def _elemental_a(symbol: str) -> float:
        """ASE の reference_states から FCC 格子定数 (Å) を返す"""
        a = reference_states[atomic_numbers[symbol]].get('a')
        if a is None:
            raise ValueError(f"No reference lattice constant for {symbol}")
        return a

    n = len(elements)
    if fractions is None:
        fractions = [1.0 / n] * n
    if abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("Fractions must sum to 1")
    
    return sum(_elemental_a(el) * frac for el, frac in zip(elements, fractions))


def make_metal_surface(num_samples=1, output_dir="./", size=[3, 3, 3],
                       elements=["Pt"], jsonfile="structures.json"):

    if len(elements) == 1:
        # --- パラメータ設定 ---
        vacuum = DEFAULT_VACUUM
        # --- 出力先ディレクトリの設定 ---
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 出力ファイルのパス（jsonを指定されたフォルダ内に出力）
        db_path = os.path.join(output_dir, jsonfile)
        if os.path.exists(db_path):
            os.remove(db_path)
        db = connect(db_path)
        # Vegard法で格子定数を計算
        lattice_const = vegard_lattice_constant(elements)
        # fcc111Bulkの作成（基本はmainの構造）
        surf = fcc111(symbol=elements[0], size=size, a=lattice_const, vacuum=vacuum, periodic=True)
        # --- EMT計算器の設定（FutureWarning解消のため calc 属性を直接代入） ---
        surf.calc = EMT()
        # --- 表面情報の取得 ---
        ads_info = surf.info["adsorbate_info"]
        # --- データベースへの書き込み ---
        data = {
            "chemical_formula": surf.get_chemical_formula(),
            "lattice_constant": float(lattice_const),
            "adsorbate_info": ads_info
        }
        db.write(surf, data=data)
        print(f"Structures saved to {db_path}")

    elif len(elements) == 2:
        db_path = _make_bimetallic_alloys(num_samples=num_samples, output_dir=output_dir,
                                          size=size, elements=elements, jsonfile=jsonfile)


def _make_bimetallic_alloys(num_samples=10, output_dir="./", size=[3, 3, 3],
                           elements=["Pt", "Ni"], jsonfile="structures.json"):
    # --- パラメータ設定 ---
    vacuum = DEFAULT_VACUUM

    # --- 出力先ディレクトリの設定 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 出力ファイルのパス（jsonを指定されたフォルダ内に出力）
    db_path = os.path.join(output_dir, jsonfile)
    if os.path.exists(db_path):
        os.remove(db_path)

    db = connect(db_path)

    # ランダムシード設定（再現性のため）
    np.random.seed(DEFAULT_RANDOM_SEED)

    # generation just to have the number of atoms in the surf
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
    return db_path


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
    
    def _filter_lines(filepath):
        """Helper function to read file and filter out comments and blank lines."""
        with open(filepath, "r") as f:
            return [line for line in f if not re.match(r'^(#|\s*$)', line)]
    
    lines = _filter_lines(file)
    numlines = len(lines)

    reac = list(range(numlines))
    rxn = list(range(numlines))
    prod = list(range(numlines))

    for i, line in enumerate(lines):
        text = line.replace("\n", "").replace(">", "").split("--")
        reac_tmp, rxn_tmp, prod_tmp = text[0], text[1], text[2]

        reac[i] = remove_space(re.split(r" \+ ", reac_tmp))
        prod[i] = remove_space(re.split(r" \+ ", prod_tmp))
        rxn[i] = reac[i][0] + "_" + rxn_tmp

    return reac, rxn, prod


def return_lines_of_reactionfile(file):
    """Return lines of reaction file."""
    import re
    
    def _filter_lines(filepath):
        """Helper function to read file and filter out comments and blank lines."""
        with open(filepath, "r") as f:
            return [line for line in f if not re.match(r'^(#|\s*$)', line)]
            
    return _filter_lines(file)


def remove_space(obj):
    """Remove space in the object."""
    if isinstance(obj, str):
        return obj.replace(" ", "")
    elif isinstance(obj, list):
        result = []
        for item in obj:
            if isinstance(item, list):
                result.append(item[-1].strip() if item else "")
            elif isinstance(item, str):
                result.append(item.replace(" ", ""))
            else:
                result.append(item)
        return result
    else:
        print("remove_space: input str or list")
        return obj


def get_reac_and_prod(reactionfile):
    """Form reactant and product information."""
    
    def _parse_molecule(mol):
        """Parse molecule string to extract coefficient, adsorbate, and site info."""
        # Extract coefficient
        if "*" in mol:
            coef = float(mol.split("*")[0])
            rest = mol.split("*")[1]
        else:
            coef = 1
            rest = mol

        # Extract sites and adsorbates
        ads_list, site_list = [], []
        if ',' in rest:
            sites = rest.split(',')
            for site in sites:
                parts = site.split('_')
                ads_list.append(parts[0])
                site_list.append(parts[1])
        elif '_' in rest:
            parts = rest.split('_')
            ads_list.append(parts[0])
            site_list.append(parts[1])
        else:
            ads_list.append(rest)
            site_list.append("gas")
        
        return coef, ads_list, site_list
    
    (reac, rxn, prod) = read_reactionfile(reactionfile)
    rxn_num = len(rxn)

    r_ads, r_site, r_coef = [], [], []
    p_ads, p_site, p_coef = [], [], []

    for irxn in range(rxn_num):
        # Process reactants
        r_mol_ads, r_mol_site, r_mol_coef = [], [], []
        for mol in reac[irxn]:
            coef, ads_list, site_list = _parse_molecule(mol)
            r_mol_coef.append(coef)
            r_mol_ads.append(ads_list)
            r_mol_site.append(site_list)
        
        r_ads.append(r_mol_ads)
        r_site.append(r_mol_site)
        r_coef.append(r_mol_coef)

        # Process products  
        p_mol_ads, p_mol_site, p_mol_coef = [], [], []
        for mol in prod[irxn]:
            coef, ads_list, site_list = _parse_molecule(mol)
            p_mol_coef.append(coef)
            p_mol_ads.append(ads_list)
            p_mol_site.append(site_list)
        
        p_ads.append(p_mol_ads)
        p_site.append(p_mol_site)
        p_coef.append(p_mol_coef)

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
    """Calculate rate coefficients using Arrhenius equation."""
    rxn_num = get_number_of_reaction(reactionfile)
    RT = 8.314 * T / 1000.0  # kJ/mol
    
    kfor = Afor * np.exp(-Efor / RT)
    krev = Arev * np.exp(-Erev / RT)
    
    return kfor, krev


def read_speciesfile(speciesfile):
    """Read species file and return list of species."""
    with open(speciesfile) as f:
        content = f.read().strip()
    
    # Remove brackets, spaces, quotes and split
    import re
    content = re.sub(r'[\[\]\s\'\"]+', '', content)
    return [s.strip() for s in content.split(',') if s.strip()]


def remove_parentheses(file):
    """Remove square brackets from file - for MATLAB compatibility."""
    with open(file, 'r') as f:
        content = f.read()
    
    # Remove square brackets
    content = content.replace('[', '').replace(']', '')
    
    with open(file, 'w') as f:
        f.write(content)
    
    return content


def get_species_num(*species):
    """Return species count or index from species.txt file."""
    from reaction_tools import read_speciesfile
    
    lst = read_speciesfile("species.txt")
    return len(lst) if not species else lst.index(species[0])


def get_adsorption_sites(infile):
    """Read adsorption sites from file."""
    mol, site = [], []
    
    with open(infile, "r") as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                mol_part, site_part = line.split(":", 1)
                mol.append(remove_space(mol_part))
                site.append(remove_space(site_part).split(","))
    
    return mol, site


def find_closest_atom(surf, offset=(0, 0)):
    """
    Find the closest atom to the adsorbate.
    """
    from ase.build import add_adsorbate

    dummy = Atom('H', (0, 0, 0))
    ads_height = DEFAULT_ADS_HEIGHT
    add_adsorbate(surf, dummy, ads_height, position=(0, 0), offset=offset)
    natoms = len(surf.get_atomic_numbers())
    last = natoms - 1
    # ads_pos = surf.get_positions(last)

    dist = surf.get_distances(last, [range(natoms)], vector=False)
    dist = np.array(dist)
    dist = np.delete(dist, last)  # delete adsorbate itself
    clothest = np.argmin(dist)

    return clothest


def sort_atoms_by_z(atoms, elementwise=True, return_zcount=False):
    """Sort atoms by z-coordinate."""
    if not atoms:
        return atoms if not return_zcount else (atoms, [])
    
    # Preserve original properties
    positions = atoms.get_positions()
    tags, pbc, cell = atoms.get_tags(), atoms.get_pbc(), atoms.get_cell()
    
    if elementwise:
        symbols = atoms.get_chemical_symbols()
        unique_elements = list(dict.fromkeys(symbols))
        sorted_atoms = Atoms()
        zcount = []
        
        for element in unique_elements:
            indices = [i for i, sym in enumerate(symbols) if sym == element]
            z_coords = positions[indices, 2]
            sorted_indices = np.argsort(z_coords)
            
            for idx in sorted_indices:
                sorted_atoms.append(atoms[indices[idx]])
            
            if return_zcount:
                rounded_z = np.round(z_coords, 2)
                zcount.append(list(Counter(rounded_z).values()))
    else:
        sorted_indices = np.argsort(positions[:, 2])
        sorted_atoms = Atoms([atoms[i] for i in sorted_indices])
        zcount = [] if return_zcount else None

    # Restore properties
    sorted_atoms.set_tags(tags)
    sorted_atoms.set_pbc(pbc)
    sorted_atoms.set_cell(cell)

    return (sorted_atoms, zcount) if return_zcount else sorted_atoms


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
    """Read charge from molecule."""
    if "^" not in mol:
        return mol, True, 0.0
    
    mol, charge_str = mol.split('^')
    charge = float(charge_str.replace('{', '').replace('}', '').replace('+', ''))
    return mol, False, charge


def remove_side_and_flip(mol):
    """Remove SIDE and FLIP suffixes from molecule."""
    suffixes = ['-SIDEx', '-SIDEy', '-SIDE', '-FLIP', '-TILT', '-HIGH']
    for suffix in suffixes:
        if suffix in mol:
            return mol.replace(suffix, '')
    return mol


def neb_copy_contcar_to_poscar(nimages):
    """Copy 0X/CONTCAR to 0X/POSCAR after NEB run."""
    import shutil
    for i in range(1, nimages + 1):
        shutil.copy2(f'{i:02d}/CONTCAR', f'{i:02d}/POSCAR')
    return True


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
    """Returns adsorbate type: 'gaseous', 'surface', or 'adsorbed'."""
    if site == "gas":
        return "surface" if "surf" in adsorbate else "gaseous"
    return "adsorbed"


def make_surface_from_cif(
        cif_file: str,
        layers: int=2,
        indices: list=[0, 0, 1],
        repeat: list=[1, 1, 1],
        vacuum: float=DEFAULT_VACUUM) -> Atoms:
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
    """Run PACKMOL with specified parameters."""
    packmol = "/Users/ishi/packmol/packmol"
    cell_coords = " ".join(map(str, [0.0, 0.0, 0.0, a, a, a]))
    
    # Write input file
    input_content = f"""tolerance 2.0
output {outfile}
filetype xyz
structure {xyz_file}
  number {num}
  inside box {cell_coords}
end structure"""
    
    with open("pack_tmp.inp", "w") as f:
        f.write(input_content)
    
    result = os.system(f"{packmol} < pack_tmp.inp")
    return result


def json_to_csv(jsonfile, csvfile):
    """Convert JSON to CSV file."""
    
    def _process_ase_json_data(jsonfile):
        """Common processing for ASE JSON files."""
        with open(jsonfile, "r") as f:
            d = json.load(f)

        dd = []
        for i in range(1, len(d)):
            if str(i) in d:
                dd.append(pd.json_normalize(d[str(i)]))

        if not dd:
            return pd.DataFrame()
            
        ddd = pd.concat(dd)

        # Clean column names
        new_columns = []
        for key in ddd.columns:
            key = key.replace("calculator_parameters.", "")
            key = key.replace("key_value_pairs.", "")
            key = key.replace("data.", "")
            new_columns.append(key)
        ddd.columns = new_columns

        # Sort by "num" if available
        if "num" in ddd.columns:
            ddd = ddd.set_index("num").sort_index()

        return ddd
    
    df = _process_ase_json_data(jsonfile)
    df.to_csv(csvfile)
    return df


def load_ase_json(jsonfile):
    """Load ASE JSON file and return as DataFrame."""
    
    def _process_ase_json_data(jsonfile):
        """Common processing for ASE JSON files."""
        with open(jsonfile, "r") as f:
            d = json.load(f)

        dd = []
        for i in range(1, len(d)):
            if str(i) in d:
                dd.append(pd.json_normalize(d[str(i)]))

        if not dd:
            return pd.DataFrame()
            
        ddd = pd.concat(dd)

        # Clean column names
        new_columns = []
        for key in ddd.columns:
            key = key.replace("calculator_parameters.", "")
            key = key.replace("key_value_pairs.", "")
            key = key.replace("data.", "")
            new_columns.append(key)
        ddd.columns = new_columns

        # Sort by "num" if available
        if "num" in ddd.columns:
            ddd = ddd.set_index("num").sort_index()

        return ddd
    
    return _process_ase_json_data(jsonfile)


def delete_num_from_json(num, jsonfile):
    from ase.db import connect

    db = connect(jsonfile)
    id_ = db.get(num=num).id
    db.delete([id_])
    return id_


def sort_atoms_by(atoms, xyz="x", elementwise=True):
    """Sort atoms by specified coordinate (x, y, z)."""
    # Preserve original properties
    tags, pbc, cell = atoms.get_tags(), atoms.get_pbc(), atoms.get_cell()
    coord_map = {"x": 0, "y": 1, "z": 2}
    coord_idx = coord_map.get(xyz, 2)
    
    def _get_coord_value(atom):
        return atom.position[coord_idx]
    
    newatoms = Atoms()
    
    if elementwise:
        symbols = list(set(atoms.get_chemical_symbols()))
        for symbol in symbols:
            subatoms = [atom for atom in atoms if atom.symbol == symbol]
            subatoms.sort(key=_get_coord_value)
            newatoms.extend(subatoms)
    else:
        sorted_atoms = sorted(atoms, key=_get_coord_value)
        newatoms.extend(sorted_atoms)

    # Restore properties
    newatoms.set_tags(tags)
    newatoms.set_pbc(pbc)
    newatoms.set_cell(cell)

    return newatoms


def get_number_of_layers(atoms):
    """Get number of z-layers for each element."""
    symbols = sorted(set(atoms.get_chemical_symbols()))
    nlayers = []

    for symbol in symbols:
        zpos = atoms.positions[[i for i, s in enumerate(atoms.get_chemical_symbols()) if s == symbol]][:, 2]
        nlayers.append(len(set(np.round(zpos, decimals=4))))

    return nlayers


def set_tags_by_z(atoms, elementwise=True):
    """Set tags based on z-coordinate layers."""
    pbc, cell = atoms.get_pbc(), atoms.get_cell()
    
    def _create_z_tags(subatoms):
        """Create z-based tags for a set of atoms."""
        zpos = np.round(subatoms.positions[:, 2], decimals=1)
        unique_z = np.sort(list(set(zpos)))
        bins = np.insert(unique_z + 1.0e-2, 0, 0)
        labels = list(range(len(bins) - 1))
        return pd.cut(zpos, bins=bins, labels=labels).tolist()
    
    newatoms = Atoms()
    
    if elementwise:
        symbols = sorted(set(atoms.get_chemical_symbols()))
        for symbol in symbols:
            subatoms = Atoms([atom for atom in atoms if atom.symbol == symbol])
            tags = _create_z_tags(subatoms)
            subatoms.set_tags(tags)
            newatoms += subatoms
    else:
        subatoms = atoms.copy()
        tags = _create_z_tags(subatoms)
        subatoms.set_tags(tags)
        newatoms += subatoms

    # Restore properties
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
    # newatoms = sort_atoms_by(newatoms, xyz="z")  # sort
    newatoms = sort_atoms_by_z(atoms=newatoms)
    newatoms = set_tags_by_z(atoms=newatoms)  # set tags

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


def find_highest(json_file, score):
    """Find entry with highest score from JSON file."""
    df = pd.read_json(json_file).set_index("unique_id")
    df = df.dropna(subset=[score]).sort_values(score, ascending=False)
    return df.iloc[0].name


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
    """Mirror invert the surface in the specified direction."""
    pos = atoms.get_positions()
    cell = atoms.cell.copy()
    
    coord_map = {"x": 0, "y": 1, "z": 2}
    if direction not in coord_map:
        raise ValueError("direction must be x, y, or z")
    
    coord_idx = coord_map[direction]
    
    if direction == "z":
        atoms.translate([0, 0, -pos[:, 2].max()])
    
    pos[:, coord_idx] = -pos[:, coord_idx]
    cell[coord_idx] = -cell[coord_idx]
    
    atoms.set_positions(pos)
    atoms.set_cell(np.round(cell + 1.0e-5, decimals=4))
    
    return atoms


def make_barplot(labels=None, values=None, threshold=100, ylabel="y-value",
                 fontsize=16, filename="bar_plot.png"):
    """Make a bar plot of values with labels, filtering by threshold."""
    import matplotlib.pyplot as plt
    
    def _style_plot_axes(ax, fontsize=16):
        """Apply common styling to plot axes."""
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(2)
        ax.xaxis.set_tick_params(direction="out", labelsize=fontsize, width=2, pad=10)
        ax.yaxis.set_tick_params(direction="out", labelsize=fontsize, width=2, pad=10)

    # Filter and sort data
    filtered_data = [(label, value) for label, value in zip(labels, values) if value < threshold]
    filtered_data.sort(key=lambda x: x[1])
    sorted_labels, sorted_values = zip(*filtered_data) if filtered_data else ([], [])

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.bar(sorted_labels, sorted_values, color="skyblue")
    
    _style_plot_axes(ax, fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize+4, labelpad=20)
    plt.xticks(rotation=45, verticalalignment="top", horizontalalignment="right")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return filename


def make_energy_diagram(deltaEs=None, has_barrier=False, rds=1, savefig=True,
                        figname="ped.png", xticklabels=None):
    """Generate potential energy diagram for reaction steps using BEP relationship."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import interpolate

    deltaEs = np.array(deltaEs)
    num_rxn = len(deltaEs)

    # Calculate cumulative energies
    ped = np.cumsum(np.insert(deltaEs, 0, 0))
    y1 = ped.copy()

    # Handle transition state barriers using BEP relationship
    if has_barrier:
        Ea = y1[rds] * BEP_ALPHA + BEP_BETA
        y1 = np.insert(y1, rds, y1[rds])
        num_rxn += 1

    # Generate interpolation
    points = 500
    x1_latent = np.linspace(-0.5, num_rxn + 0.5, points)
    x1 = np.arange(0, num_rxn + 1)
    f1 = interpolate.interp1d(x1, y1, kind="nearest", fill_value="extrapolate")

    # Create barrier curve if needed
    if has_barrier:
        x2 = np.array([rds - 0.5, rds, rds + 0.5])
        y2 = np.array([y1[rds-1], Ea, y1[rds+1]])
        f2 = interpolate.interp1d(x2, y2, kind="quadratic")
        y = np.array([max(f1(i), f2(i) if rds-0.5 <= i <= rds+0.5 else -1e10) for i in x1_latent])
    else:
        y = f1(x1_latent)

    if savefig:
        sns.set_theme(style="darkgrid", rc={"lines.linewidth": 2.0, "figure.figsize": (10, 4)})
        p = sns.lineplot(x=x1_latent, y=y)
        p.set_xlabel("Steps", fontsize=16)
        p.set_ylabel("Energy (eV)", fontsize=16)
        p.tick_params(axis="both", labelsize=14)
        p.yaxis.set_major_formatter(lambda x, p: f"{x:.1f}")
        
        if xticklabels:
            xticklabels.insert(0, "dummy")
            p.set_xticklabels(xticklabels, rotation=45, ha="right")
        
        plt.tight_layout()
        plt.savefig(figname)

    return x1_latent, y


def add_data_to_jsonfile(jsonfile, data):
    """Add data to JSON database file."""
    # Initialize file if it doesn't exist
    if not Path(jsonfile).exists():
        with open(jsonfile, "w") as f:
            json.dump([], f)

    # Read, modify, and write back
    with open(jsonfile, "r") as f:
        datum = json.load(f)
    
    # Remove any "doing" status records
    datum = [d for d in datum if d.get("status") != "doing"]
    datum.append(data)

    with open(jsonfile, "w") as f:
        json.dump(datum, f, indent=4)
    
    return len(datum)


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


def set_initial_magmoms(atoms: Atoms) -> Atoms:
    """Set initial magnetic moments based on element type."""
    magmoms = [MAGMOM_DICT.get(symbol, 0.0) for symbol in atoms.get_chemical_symbols()]
    atoms.set_initial_magnetic_moments(magmoms)
    return atoms


def set_calculator(atoms: Atoms, kind: str, calculator: str = "mace",
                   yaml_path: str = None, calc_directory: str = "calc"):
    """
    Create calculator instance based on parameters from YAML file and attach to atoms.

    Args:
        atoms: ASE atoms object
        kind: "molecule" / "surface" / "solid"
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

    if calculator == "vasp":
        from ase.calculators.vasp import Vasp

        if yaml_path is None:
            # load default yaml file
            yaml_path = Path(__file__).resolve().parent / "data" / "vasp_default.yaml"

        # Load YAML file directly
        try:
            with open(yaml_path, "r") as f:
                vasp_params = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: VASP parameter file not found at {yaml_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {yaml_path}: {e}")
            sys.exit(1)

        if kind not in vasp_params["kinds"]:
            raise ValueError(f"Invalid kind '{kind}'. Must be one of {list(vasp_params['kinds'].keys())}")

        # Copy common parameters
        params = vasp_params["common"].copy()

        # Update with kind-specific parameters
        params.update(vasp_params["kinds"][kind])

        # Set function argument parameters
        params["directory"] = calc_directory

        # Convert kpts to tuple (ASE expects tuple)
        if "kpts" in params and isinstance(params["kpts"], list):
            params["kpts"] = tuple(params["kpts"])

        # Set calculator to atoms object and return
        atoms.calc = Vasp(**params)

        # Automatically set lmaxmix
        atoms = auto_lmaxmix(atoms)

    elif calculator == "mattersim":
        from mattersim.forcefield.potential import MatterSimCalculator
        from ase.filters import ExpCellFilter

        device = "cuda" if torch.cuda.is_available() else "cpu"
        atoms.calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)

        # Apply CellFilter for bulk calculations
        if kind == "bulk":
            atoms = ExpCellFilter(atoms)

        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        else:
            atoms = atoms

    elif calculator == "mattersim-matpes-pbe-d3":
        # Import the custom function
        from mattersim_matpes import mattersim_matpes_d3_calculator
        from ase.filters import ExpCellFilter

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

        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        else:
            atoms = atoms

    elif calculator == "mattersim-matpes-pbe":
        # Import the custom function
        from mattersim_matpes import mattersim_matpes_d3_calculator
        from ase.filters import ExpCellFilter

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

        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        else:
            atoms = atoms

    elif calculator == "mace":
        from mace.calculators import mace_mp
        from ase.filters import ExpCellFilter

        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/"
        # model = "MACE-matpes-pbe-omat-ft.model"
        model = "MACE-matpes-r2scan-omat-ft.model"

        mace_calculator = mace_mp(model=base_url + model,
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

                # Handle special methods used by deepcopy
                if name.startswith('__') and name.endswith('__'):
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
                try:
                    return getattr(self._calculator, name)
                except AttributeError:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # 保護されたカリキュレータをセット
        atoms.calc = ProtectedMaceCalculator(mace_calculator)

        # Apply CellFilter for bulk calculations
        if kind == "bulk":
            atoms = ExpCellFilter(atoms)

        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        else:
            atoms = atoms

    else:
        raise ValueError("calculator must be 'vasp' or 'mace'")

    return atoms


def auto_lmaxmix(atoms):
    """Automatically set lmaxmix when d/f elements are present"""
    symbols = set(atoms.get_chemical_symbols())

    if symbols & F_ELEMENTS:
        lmaxmix_value = 6
    elif symbols & D_ELEMENTS:
        lmaxmix_value = 4
    else:
        lmaxmix_value = 2

    atoms.calc.set(lmaxmix=lmaxmix_value)
    return atoms


def plot_free_energy_diagram(deltaGs: List[float], steps: List[str], work_dir: Path):
    """Plot free energy diagram."""
    if steps is None:
        raise ValueError("in plot_free_energy_diagram: steps are not given")

    import matplotlib.pyplot as plt

    cumulative_energies = [0] + np.cumsum(deltaGs).tolist()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(steps)), cumulative_energies, 'o-', linewidth=2, markersize=8)
    plt.xlabel("Reaction Step")
    plt.ylabel("Free Energy (eV)")
    plt.title("ORR Free Energy Diagram")
    plt.xticks(range(len(steps)), steps, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(work_dir / "free_energy_diagram.png", dpi=300, bbox_inches='tight')
    plt.close()
    return str(work_dir / "free_energy_diagram.png")


def convert_numpy_types(obj):
    """Convert NumPy types to standard Python types"""
    import numpy as np

    if isinstance(obj, np.number):
        return obj.item()  # Convert NumPy numeric types to Python standard numeric types
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj


def parallel_displacement(atoms, vacuum=DEFAULT_VACUUM):
    """Translate surface in z-direction so the lowest point becomes z=0.
    
    Args:
        atoms: ASE Atoms object (surface)
        vacuum: Thickness of vacuum layer to add (Å)
    """
    # Create copy to avoid modifying original object
    surf = atoms.copy()

    # Get current atomic positions and calculate minimum z value
    positions = surf.get_positions()
    z_min = positions[:, 2].min()

    # Translate entire surf in z-direction so lowest point becomes z=0
    surf.translate([0, 0, -z_min])

    # Get maximum z coordinate after translation
    z_max = surf.get_positions()[:, 2].max()
    # Calculate new z-axis length (surf height + vacuum)
    new_z_length = z_max + vacuum

    # Get cell matrix and set z-direction size to new length
    # Here we assume the cell's third vector aligns with z-direction
    cell = surf.get_cell().copy()
    # For safety, reset z-axis component to [0, 0, new_z_length]
    cell[2] = [0.0, 0.0, new_z_length]
    surf.set_cell(cell, scale_atoms=False)  # scale_atoms=False updates only cell, not atomic coordinates

    return surf


def plot_energy_diagram(steps, values, color='blue', labels=None,
                        label=None, line_width=0.3, alpha=1.0, marker='o',
                        markersize=5, figname=None):
    """Helper function to plot a energy profile with consistent styling."""
    import matplotlib.pyplot as plt

    # Plot horizontal lines
    for i, step in enumerate(steps):
        line_label = label if i == 0 else None
        plt.hlines(values[i], step - line_width, step + line_width,
                   color=color, alpha=alpha, linewidth=2.5, label=line_label)

    # Connect with dashed lines
    for i in range(len(steps) - 1):
        plt.plot([steps[i] + line_width, steps[i + 1] - line_width],
                 [values[i], values[i + 1]], '--', color=color, alpha=alpha, linewidth=1.0)

    plt.xticks(steps, labels, rotation=45, ha='right')
    plt.tight_layout()

    # Add markers
    plt.plot(steps, values, marker, color=color, alpha=alpha,
             markersize=markersize, linestyle='none')

    if figname is not None:
        plt.savefig(figname)


def make_plot(labels=None):
    """Generate ORR free energy diagram with multiple potential profiles."""
    import matplotlib.pyplot as plt

    # Define reaction step labels and data
    if labels is None:
        labels = ["O$_2$ + 2H$_2$", "OOH* + 1.5H$_2$", "O* + H$_2$O + H$_2$",
                  "OH* + H$_2$O + 0.5H$_2$", "* + 2H$_2$O"]
    
    steps = np.arange(reaction_count + 1)
    profiles = {
        'U=0V': (g_profile_u0 - g_profile_u0[-1], 'black', 0.6, 'o', 4),
        f'U_L={limiting_potential:.2f}V': (g_profile_ul - g_profile_ul[-1], 'blue', 1.0, 's', 5),
        f'U={equilibrium_potential}V': (g_profile_ueq - g_profile_ueq[-1], 'green', 0.8, 'o', 6)
    }
    
    line_width = 0.3
    plt.figure(figsize=(8, 7))
    
    # Plot each profile
    for label, (profile_data, color, alpha, marker, markersize) in profiles.items():
        plot_energy_diagram(steps, profile_data, color, label, line_width, alpha, marker, markersize)
    
    # Formatting
    plt.xticks(steps, labels, rotation=15, ha='right')
    plt.ylabel("ΔG (eV)", fontsize=12, fontweight='bold')
    plt.xlabel("Reaction Coordinate", fontsize=12, fontweight='bold') 
    plt.title("4e⁻ ORR Free-Energy Diagram", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ORR_free_energy_diagram.png", dpi=300, bbox_inches='tight')
    plt.close()

    return str(output_dir / "ORR_free_energy_diagram.png")
