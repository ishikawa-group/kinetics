from kinetics.hydration import get_hydration_energy

cif = "BaZrO3.cif"
hydration, vacancy, oh = get_hydration_energy(cif)
print(f"hydration energy = {hydration:5.3f} eV")
print(f"oxygen vacation formation energy = {vacancy:5.3f} eV")
print(f"OH formation energy = {oh:5.3f} eV")
