def get_hydration_energy(cif_file=None):
    """
    Calculates and prints defect formation and hydration energies with ZPE corrections.

    Args:
        cif_file (str): Path to the CIF file.
    Returns:
        delta_E_hydr (float): Hydration energy.
        delta_E_VOx (float): Oxygen vacancy formation energy.
        delta_E_OHx (float): OH adsorption energy.
    """
    import os
    from ase.io import read
    from ase import Atom
    from ase.io.vasp import read_vasp_out
    from ase.calculators.vasp import Vasp

    # constant energy values for H2O, H2, O2 (eV)
    E_H2O = -14.891
    E_H2 = -6.715
    E_O2 = -9.896

    # constant ZPE values (eV)
    ZPE_H2O = 0.56
    ZPE_H2 = 0.27
    ZPE_O2 = 0.10

    # apply ZPE corrections
    E_H2_corr = E_H2 + ZPE_H2
    E_H2O_corr = E_H2O + ZPE_H2O
    E_O2_corr = 2 * ((E_H2O + ZPE_H2O) - (E_H2 + ZPE_H2)) - ZPE_O2

    # read the CIF file
    atoms = read(cif_file)

    # make the hydrated structure
    atoms_with_Vox = atoms.copy()

    # Step1: Remove an oxygen atom to simulate the vacancy(Vox)

    # Find the oxygen atom 
    oxygen_indices = [atom.index for atom in atoms_with_Vox if atom.symbol == "O"]
    # Record the oxygen atom position
    oxygen_position = atoms[oxygen_indices[0]].position
    # Remove an oxygen atom
    minus_O = atoms_with_Vox.pop(oxygen_indices[0])

    # Step2: Add back the oxygen atom and a hydrogen atom to simulate proton defect(OHx)
    # Copy the structure with the vacancy
    atoms_with_OHx = atoms_with_Vox.copy()
    # Add an oxygen atom
    atoms_with_OHx.append(Atom("O", position=oxygen_position))
    # Add a hydrogen atom
    atoms_with_OHx.append(Atom("H", position=oxygen_position + [0.0, 0.0, 0.96]))

    # Do VASP calculation in this script or not
    calculate_here = True

    if calculate_here:
        tmpdir = "tmpdir_hydration"
        tmpdir_vox = "tmpdir_hydration_vox"
        tmpdir_ohx = "tmpdir_hydration_ohx"

        atoms.calc = Vasp(prec="normal", xc="pbe", ibrion=-1, nsw=0, directory=tmpdir)
        atoms_with_Vox.calc = Vasp(prec="normal", xc="pbe", ibrion=-1, nsw=0, directory=tmpdir_vox)
        atoms_with_OHx.calc = Vasp(prec="normal", xc="pbe", ibrion=-1, nsw=0, directory=tmpdir_ohx)

        E_pristine = atoms.get_potential_energy()
        E_Vox = atoms_with_Vox.get_potential_energy()
        E_OHx = atoms_with_OHx.get_potential_energy()

    else:
        # Ensure the OUTCAR files exist
        for dir_name in ['pristine', 'Vox', 'OHx']:
            if not os.path.exists(f'{dir_name}/OUTCAR'):
                raise FileNotFoundError(f"OUTCAR not found in {dir_name} directory. Make sure VASP has finished running.")
    
        # Retrieve the energies from the OUTCAR files
        atoms_pristine = read_vasp_out('pristine/OUTCAR')
        E_pristine = atoms_pristine.get_potential_energy()

        atoms_Vox = read_vasp_out('Vox/OUTCAR')
        E_Vox = atoms_Vox.get_potential_energy()

        atoms_OHx = read_vasp_out('OHx/OUTCAR')
        E_OHx = atoms_OHx.get_potential_energy()

    # calculate hydration energy
    delta_E_Vox = E_Vox - (E_pristine + 0.5*E_O2_corr)
    delta_E_OHx = E_OHx - (E_pristine + (E_H2O_corr - E_H2_corr*0.5))
    delta_E_hydr = (2*E_OHx) - (E_pristine + E_Vox + E_H2O_corr)

    # return results
    return delta_E_hydr, delta_E_Vox, delta_E_OHx

if __name__ == "__main__":
    get_hydration_energy("../BaZrO3.cif")
