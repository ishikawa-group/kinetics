from mp_api.client import MPRester
import os

MY_API_KEY = os.environ["MAPI"]

# Get cif files of ABO3 systems
name = "**O3"
band_gap_min = 3.0  # Only for stable insulators with band_gap of this value
band_gap_max = None
sg_symb = "Pm-3m"   # spacegroup = Pm-3m
_is_stable = None
_is_metal  = False

# define physical properties/infos you want to obtain
properties = ["formula_pretty", "material_id", "structure", "symmetry", "is_metal", "band_gap"]
with MPRester(MY_API_KEY) as mpr:
    results = mpr.materials.summary.search(formula=name, band_gap=(band_gap_min, band_gap_max),
                                           spacegroup_symbol=sg_symb, is_stable=_is_stable, is_metal=_is_metal,
                                           fields=properties)

# Output
output_dir = "ABO3_cif"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for mat in results:
    print(f"formula = {mat.formula_pretty:>8.6s}, is_metal = {str(mat.is_metal):6.5s}, bandgap = {mat.band_gap:5.3f}")
    filename = mat.material_id + "_"+ mat.formula_pretty + ".cif"
    filename = os.path.join(output_dir, filename)
    mat.structure.to(filename=filename)
    print(f"{len(results)} materials are found.")
