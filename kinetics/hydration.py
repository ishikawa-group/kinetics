def set_vasp_calculator(atoms=None, directory=None):
    """
    Set VASP calculator for Atoms.

    Args:
        atoms: ASE Atoms object.
    """
    from ase.calculators.vasp import Vasp

    calc = Vasp(prec="normal", xc="pbe", ispin=2, lorbit=10,
                encut=520, ediff=1e-6, algo="Normal", nelm=40, nelmin=5,
                ibrion=2, nsw=50, isif=2, potim=0.1, ediffg=-0.05,
                ismear=0, sigma=0.05,
                kpts=[4, 4, 4], kgamma=True,
                npar=4, nsim=4,
                directory=directory,
                lwave=False, lcharg=False,
                lreal=False,
                )
    atoms.calc = calc
    return None


def get_molecule_energy(species=None):
    """
    """
    from ase.build import molecule
    from ase.calculators.vasp import Vasp

    mol = molecule(species)
    mol.cell = [20]*3
    mol.pbc = True
    mol.center()

    ispin = 2 if species == "O2" else 1

    directory = "tmpdir_" + species
    mol.calc = Vasp(prec="normal", xc="pbe", encut=400, ispin=ispin, ibrion=2, nsw=50, ismear=0, kpts=[1,1,1], directory=directory)
    energy = mol.get_potential_energy()

    return energy


def get_hydration_energy(atoms=None):
    """
    Calculates and prints defect formation and hydration energies with ZPE corrections.

    Args:
        atoms: ASE Atoms object.
    Returns:
        delta_E_hydr (float): Hydration energy
        delta_E_VOx (float): Oxygen vacancy formation energy
        delta_E_OHx (float): Proton formation energy
    """
    import os
    from ase.io import read
    from ase import Atom
    from ase.io.vasp import read_vasp_out
    from ase.calculators.vasp import Vasp

    # constant energy values for H2O, H2, O2 (eV)
    # E_H2O = -14.891
    # E_H2 = -6.715
    E_H2  = get_molecule_energy(species="H2")
    E_H2O = get_molecule_energy(species="H2O")

    # constant ZPE values (eV)
    ZPE_H2O = 0.56
    ZPE_H2 = 0.27
    ZPE_O2 = 0.10

    # apply ZPE corrections
    E_H2_corr = E_H2 + ZPE_H2
    E_H2O_corr = E_H2O + ZPE_H2O

    # H2O formation enthalpy (experimental value, 241.8 (gas) or 285.8 (liquid) kJ/mol -> eV)
    E_exp_HOF = -2.507  # -2.962

    E_O2_corr = 2 * ((E_H2O + ZPE_H2O) - (E_H2 + ZPE_H2) - E_exp_HOF) - ZPE_O2

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
    atoms_with_OHx.append(Atom("H", position=oxygen_position + [0.6, 0.0, 0.6]))

    # Do VASP calculation in this script or not
    calculate_here = True

    if calculate_here:
        tmpdir = "tmpdir_hydration"
        tmpdir_vox = "tmpdir_hydration_vox"
        tmpdir_ohx = "tmpdir_hydration_ohx"

        set_vasp_calculator(atoms, directory=tmpdir)
        atoms.calc.set(isif=8)
        set_vasp_calculator(atoms_with_Vox, directory=tmpdir_vox)
        set_vasp_calculator(atoms_with_OHx, directory=tmpdir_ohx)

        E_pristine = atoms.get_potential_energy()
        atoms_with_Vox.cell = atoms.cell
        atoms_with_OHx.cell = atoms.cell

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

    # calculate chemical potential
    mu_O = 0.5*E_O2_corr
    mu_H2O = E_H2O_corr
    mu_H = 0.5*(mu_H2O - mu_O)

    # calculate hydration energy
    delta_E_Vox = E_Vox - (E_pristine - mu_O)
    delta_E_OHx = E_OHx - (E_pristine + mu_H)
    delta_E_hydr = 2*E_OHx - (E_pristine + E_Vox + mu_H2O)

    # return results
    return delta_E_hydr, delta_E_Vox, delta_E_OHx

