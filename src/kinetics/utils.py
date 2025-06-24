import os
import argparse
import uuid
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.db import connect
from ase.data import atomic_numbers
import numpy as np


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
