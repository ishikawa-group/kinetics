import os
from ase.io import read, write
from ase.calculators.vasp import Vasp
from ase.dft.bandgap import bandgap

def calc_bandgap(cif_file=None):
    """
    Calculate band gap.

    Parameters:
        cif_file (str): Path to the CIF file of the material.

    Returns:
        float: Calculated band gap value.
    """

    # Check VASP command environment variable is set

    if os.getenv("ASE_VASP_COMMAND"):
        print("ASE_VASP_COMMAND is set to:", os.getenv("ASE_VASP_COMMAND"))
    elif os.getenv("VASP_COMMAND"):
        print("VASP_COMMAND is set to:", os.getenv("VASP_COMMAND"))
    elif os.getenv("VASP_SCRIPT"):
        print("VASP_SCRIPT is set to:", os.getenv("VASP_SCRIPT"))
    else:
        raise EnvironmentError("One of ASE_VASP_COMMAND, VASP_COMMAND, VASP_SCRIPT should be set. Please ensure it is correctly set in your environment.")

    # Check if VASP_PP_PATH is set
    if not os.getenv("VASP_PP_PATH"):
        raise EnvironmentError("VASP_PP_PATH is not set. Please ensure it is correctly set in your environment.")
    print("VASP_PP_PATH is set to:", os.getenv("VASP_PP_PATH"))

    # Verify CIF file exists
    if cif_file is None or not os.path.exists(cif_file):
        raise FileNotFoundError(f"The specified CIF file '{cif_file}' does not exist.")
    print(f"Using CIF file: {cif_file}")

    # Read the structure from the CIF file
    try:
        structure = read(cif_file)
        print("Structure read successfully from CIF file.")
    except Exception as e:
        raise RuntimeError(f"Failed to read structure from CIF file '{cif_file}': {e}")


    # Set up VASP calculator with standard settings

    directory = "test"

    try:
        structure.calc = Vasp(prec="normal",
                              xc="pbe",
                              encut=500,
                              kpts=[8, 8, 8],
                              ismear=0,
                              sigma=0.05,
                              setups='recommended',
                              lwave=True,
                              lcharg=True,
                              nelm=150,
                              algo="fast",
                              ediff=1e-8,
                              npar=4,
                              directory=directory,
                              )
        print("VASP calculator set up successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to set up VASP calculator: {e}")

    # Calculate total energy (needed for band structure calculation)
    try:
        energy = structure.get_potential_energy()
        print(f"Total Energy = {energy:.3f} eV")
    except Exception as e:
        raise RuntimeError(f"Failed to calculate total energy: {e}")

    # Calculate band gap
    try:
        band_gap, p1, p2 = bandgap(structure.calc, direct=True)
        print(f"Band Gap = {band_gap:.3f} eV, p1 = {p1}, p2 = {p2}")
    except Exception as e:
        raise RuntimeError(f"Failed to calculate band gap: {e}")
    
    return band_gap

if __name__ == "__main__":
    calc_bandgap("../BaZrO3.cif")
