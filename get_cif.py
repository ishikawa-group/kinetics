from mp_api.client import MPRester
import os

MY_API_KEY = os.environ["MAPI"]

#Get cif files of ABO3 systems
# Only for stable insulators with band_gap > 2 and spacegroup = Pm-3m
band_gap_min = 2.0
band_gap_max = None
sg_symb = "Pm-3m"
_is_stable = True
_is_metal = False

# define physical properties/infos you want to obtain
properties = ['formula_pretty','material_id','structure','symmetry','is_metal','band_gap']
name = "**O3"
with MPRester( MY_API_KEY ) as mpr:
    results = mpr.materials.summary.search(formula=name, band_gap=(band_gap_min, band_gap_max), spacegroup_symbol = sg_symb, is_stable=_is_stable, is_metal=_is_metal, fields=properties)

#Output
path_output_dir = "./ABO3_cif/"
if not os.path.exists(path_output_dir):
    os.makedirs(path_output_dir)

for mat in results:
    print(mat.material_id, mat.formula_pretty)
    print(mat.symmetry.symbol)
    print(mat.is_metal, mat.band_gap)

    ofile = path_output_dir+mat.material_id+"_"+mat.formula_pretty+".cif"
    mat.structure.to(filename = ofile)
