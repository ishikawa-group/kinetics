import os
from ase.build import bulk, surface, add_vacuum
from ase.db import connect
from ase.visualize import view

# surf = make_metal_surface(indices=[0, 0, 0, 1], size=[2, 2, 4], elements=["Co"], jsonfile="structures.json")

indices = [1, 1, 1]
size = [3, 3, 4]
vacuum = 8.0
element = "Ni"
crystalstructure = "fcc"
orthorhombic = False
a = 3.52  # fcc Ni --- 3.52, fcc Co --- 3.54

bulk = bulk(name=element, crystalstructure=crystalstructure, a=a, orthorhombic=orthorhombic)
surf = surface(lattice=bulk, indices=indices, layers=size[2], vacuum=vacuum, periodic=True)
surf = surf * [size[0], size[1], 1]
surf.translate([0, 0, -vacuum + 0.1])
surf.wrap(eps=1.0e-3)

# 出力ファイルのパス（jsonを指定されたフォルダ内に出力）
db_path = "structures.json"
if os.path.exists(db_path):
    os.remove(db_path)
db = connect(db_path)

data = {
    "chemical_formula": surf.get_chemical_formula(),
}
db.write(surf, data=data)
print(f"Structures saved to {db_path}")
