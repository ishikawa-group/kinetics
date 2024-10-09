def get_hydration_energy(E_pristine, E_Vox, E_OHx):
    """ Calculates and prints defect formation and hydration energies with ZPE corrections. """
    # Constant energy values for H2O, H2, O2 (eV)
    E_H2O = -14.891
    E_H2 = -6.715
    E_O2 = -9.896

    # Constant ZPE values (eV)
    ZPE_H2O = 0.56
    ZPE_H2 = 0.27
    ZPE_O2 = 0.10

    # Apply ZPE corrections
    E_O2_corr = 2 * ((E_H2O + ZPE_H2O) - (E_H2 + ZPE_H2)) - ZPE_O2
    E_H2_corr = E_H2 + ZPE_H2
    E_H2O_corr = E_H2O + ZPE_H2O

    # Calculate defect formation energies
    delta_E_Vox = E_Vox - E_pristine + E_O2_corr / 2
    delta_E_OHx = E_OHx - E_pristine - ((E_H2O_corr - E_H2_corr / 2) / 2)
    delta_E_hydr = 2 * E_OHx - E_pristine - E_Vox - E_H2O_corr

    # Return results
    return delta_E_Vox, delta_E_OHx, delta_E_hydr
