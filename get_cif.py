from mp_api.client import MPRester
import os

os.environ["MPRESTER_MUTE_PROGRESS_BARS"] = "true"
MY_API_KEY = os.environ["MAPI"]

# Get cif files of ABO3 systems
name = "**O3"
spacegroup = "Pm-3m"   # spacegroup = Pm-3m
_is_stable = None
_is_metal  = None  # a lot of missing data? ... will be skiped

# define physical properties/infos you want to obtain
properties = ["formula_pretty", "material_id", "structure", "symmetry", "is_metal", "band_gap"]

with MPRester(MY_API_KEY) as mpr:
    results = mpr.materials.summary.search(formula=name, spacegroup_symbol=spacegroup, 
                                           is_stable=_is_stable, is_metal=_is_metal, fields=properties)

# Output

def include_this(material, include_elements):
    include = False
    for element in include_elements:
        if element in material.formula_pretty:
            include = True
            break

    return include

def skip_this(material, skip_elements):
    skip = False
    for element in skip_elements:
        if element in material.formula_pretty:
            skip = True
            break

    return skip

skip_elements = ["Ho", "Hf", "Pa", "Th", "Ac", "Er", "Lu", "Tm", "Pr", "Sm", "Dy", "Pm", "Eu", "U", "Pu"]
include_elements = ["Mn"]

output_dir = "ABO3_cif"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

after = []
for material in results:
    # if skip_this(material, skip_elements):
    if include_this(material, include_elements):
        continue
    else:
        after.append(material)

for material in after:
    filename = material.material_id + "_"+ material.formula_pretty + ".cif"
    filename = os.path.join(output_dir, filename)
    material.structure.to(filename=filename)

print(f"{len(results)} materials are found, and {len(after)} materials are saved.")

