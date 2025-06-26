# kinetics
* Assemble of codes related to the chemical kinetics.

## Diffusion
* Please see `README.md` in the `diffusion` directory.

## Microkinetics
### Oxygen reduction reaction (ORR)
* Calculates the overpotential for oxygen reduction reaction (ORR).
* Includes zero-point energy (ZPE) and entropy corrections.

### Usage
```python
from kinetics.microkinetics.orr_and_oer import get_overpotential_oer_orr

# Reaction energies for each step (deltaEs)
# 1. O2 + H+ + e- → OOH*
# 2. OOH* + H+ + e- → O* + H2O
# 3. O* + H+ + e- → OH*
# 4. OH* + H+ + e- → H2O
deltaEs = [1.0, 0.5, 0.3, 0.2]  # example values in eV

overpotential = get_overpotential_oer_orr(
    reaction_file="path/to/reaction.txt",
    deltaEs=deltaEs,
    T=298.15,  # temperature in K
    reaction_type="orr",  # "orr" or "oer"
    verbose=True  # prints detailed energy information
)
print(f"ORR overpotential = {overpotential:5.3f} V")
```

### Oxygen evolution reaction (OER)
* Calculates the overpotential for oxygen evolution reaction (OER).
* Automatically generates surface models from CIF files and computes electronic descriptors.

#### Code Structure (`examples/oer/oer.py`)

##### Key Components:

1. **Electronic Configuration Analysis**
   - Loads electron configuration data from `electron_numbers.yaml`
   - Calculates s, p, d, f electrons for surface elements
   - Function: `get_spdf_electrons(surface: Atoms) -> tuple[int]`

2. **Surface Analysis Functions**
   - `get_min_metal_oxygen_distance(surface: Atoms) -> float`: Calculates minimum metal-oxygen bond distance
   - `make_surface_from_cif()`: Creates surface slab from bulk CIF file (imported from utils)

3. **Data Processing**
   - `write_to_csv()`: Appends calculation results to CSV file
   - Generates bar plot visualization of overpotentials

4. **Main Calculation Loop**
   - Processes CIF files from specified directories (`ABO3_cif_large/`, `ABO3_cif/`, `ABO3_Mn_cif/`)
   - Calculates OER overpotential using various calculators (MACE or VASP)
   - Extracts material descriptors and saves results

##### Parameters
```python
repeat = [2, 2, 2]        # Surface supercell size
vacuum = 7.0              # Vacuum thickness (Å)
max_sample = 5            # Maximum number of materials to process
calculator = "mace"       # Energy calculator ("vasp", "m3gnet", "mattersim")
reaction_file = "oer.txt" # OER reaction pathway definition
```

##### Output Data
* The script generates `output.csv` with columns:
- `formula`: Chemical formula of the surface
- `cell_volume`: Bulk unit cell volume
- `s_electrons`, `p_electrons`, `d_electrons`, `f_electrons`: Electronic descriptors
- `min_M_O_distance`: Minimum metal-oxygen distance
- `overpotential_in_eV`: Calculated OER overpotential

#### Usage Example:
```bash
cd examples/oer/
python oer.py
```

##### Required Files:
- `electron_numbers.yaml`: Electronic configuration database
- `oer.txt`: OER reaction pathway definition
- CIF files in subdirectories (`ABO3_cif_large/`, etc.)

##### OER Reaction Pathway (`oer.txt`):
```
H2O + surf       --ads--> OH_fcc + 0.5*H2
OH_fcc           --LH-->  O_fcc + 0.5*H2
O_fcc + H2O      --LH-->  OOH_fcc + 0.5*H2
OOH_fcc          --LH-->  O2 + surf + 0.5*H2
```

This represents the four-step OER mechanism:
1. Water adsorption and dissociation
2. OH dehydrogenation to O
3. Water oxidation to form OOH
4. O2 evolution

## Authors
* Yang Long (https://github.com/Long-Brian-Yang)
* Atsushi Ishikawa (https://github.com/atsushi-ishikawa)

