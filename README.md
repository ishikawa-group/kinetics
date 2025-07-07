# kinetics
* Assemble of codes related to the chemical kinetics.

---

## Diffusion
* Please see `README.md` in the `diffusion` directory.

---

## Microkinetics
### Example cases
* These example codes are stored in `examples` directory.

### 1. Oxygen evolution reaction (OER)
* Calculates the overpotential for OER.
* Automatically generates surface models from CIF files and computes electronic descriptors.
* Example code:`examples/oer/oer.py`

#### Key parameters
* Important parameters are as follows, in the above example code.
```python
repeat = [2, 2, 2]         # Surface supercell size
vacuum = 7.0               # Vacuum thickness (Å)
max_sample = 5             # Maximum number of materials to process
calculator = "mace"        # Energy calculator ("vasp", "m3gnet", "mattersim")
reaction_file = "oer.txt"  # OER reaction pathway definition
```

#### Output Data
* The script generates `output.csv` with columns:
  - `formula`: Chemical formula of the surface
  - `cell_volume`: Bulk unit cell volume
  - `s_electrons`, `p_electrons`, `d_electrons`, `f_electrons`: Electronic descriptors
  - `min_M_O_distance`: Minimum metal-oxygen distance
  - `overpotential_in_eV`: Calculated OER overpotential

#### Required Files:
* CIF files (single or multiple).
* `oer.txt`: OER reaction pathway definition
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


### 2. Oxygen reduction reaction (ORR)
* Similar to the OER, the library provides function to calculate the overpotential of ORR.
* Example code is `examples/orr/orr.py`.

#### Usage
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

### 3. NH3 decomposition
* Calculates the rate (TOF) of NH3 decomposition.
* Example code: `examples/nh3decomposition/nh3decomp.py`.

---

## Authors
* Yang Long (https://github.com/Long-Brian-Yang)
* Atsushi Ishikawa (https://github.com/atsushi-ishikawa)
