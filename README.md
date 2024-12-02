# kinetics
* Assemble of codes related to the chemical kinetics.

## Bandgap
* Calculates the bandgap with VASP.

### Usage
```python
from ase.io import read
from kinetics.bandgap import get_bandgap

cif = "BaZrO3.cif"
atoms = read(cif)

bandgap = get_bandgap(atoms=atoms)
print(f"bandgap = {bandgap:5.3f} eV")
```

## Hydration
* Calculates following three properties:
  1. oxygen atom vacancy formation energy
  2. H atom adsorption energy (in bulk)
  3. hydration energy

### Usage
```python
from ase.io import read
from kinetics.hydration import get_hydration_energy

cif = "BaZrO3.cif"
atoms = read(cif)

hydration, vacancy, oh = get_hydration_energy(atoms)
print(f"hydration energy = {hydration:5.3f} eV")
print(f"oxygen vacation formation energy = {vacancy:5.3f} eV")
print(f"proton formation energy = {oh:5.3f} eV")
```

## Diffusion
* Please see `README.md` in the `diffusion` directory.
