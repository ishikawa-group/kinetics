# kinetics
* Assemble of codes related to the chemical kinetics.

## Diffusion
* Please see `README.md` in the `diffusion` directory.

## Oxygen reduction reaction (ORR)
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
