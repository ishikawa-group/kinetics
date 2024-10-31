import os
from kinetics.hydration import get_hydration_energy
from kinetics.bandgap import get_bandgap

cif = "BaZrO3.cif"
bandgap = get_bandgap(cif)
print(f"bandgap = {bandgap:5.3f} eV")

hydration, vacancy, oh = get_hydration_energy(cif)
print(f"hydration energy = {hydration:5.3f} eV")
print(f"oxygen vacation formation energy = {vacancy:5.3f} eV")
print(f"OH formation energy = {oh:5.3f} eV")
