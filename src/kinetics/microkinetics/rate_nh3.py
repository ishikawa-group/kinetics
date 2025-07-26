import numpy as np
from kinetics.utils import get_number_of_reaction


def get_nh3_formation_rate(deltaEs=None, reaction_file=None, rds=0, T=300, debug=False):
    """Calculate the NH3 formation rate from reaction energies (deltaEs)."""
    p_tot = 1.0e5   # pressure [Pa]
    p_ref = 1.0e5   # pressure of standard state [Pa]
    kJtoeV = 1/98.415
    RT = 8.314*1.0e-3*T*kJtoeV  # gas constant times temperatuer [J/K -> kJ/K * K -> eV]

    gas = {"N2": 0, "H2": 1, "NH3": 2}
    ads = {"N" : 0, "H": 1, "NH": 2, "NH2": 3, "NH3": 4, "vac": 5}

    rxn_num = get_number_of_reaction(reaction_file)

    p = np.zeros(len(gas))
    conversion = 0.10
    p[gas["NH3"]] = p_tot
    p[gas["N2"]]  = p[gas["NH3"]]*conversion*(1.0/4.0)
    p[gas["H2"]]  = p[gas["NH3"]]*conversion*(3.0/4.0)
    p = p / p_ref  # divide by reference pressure (p/p_0)

    """
    zpe should be included in the deltaEs
    
    zpe = np.zeros(len(gas) + len(ads))
    zpe[gas["N2"]]  = 0.145
    zpe[gas["H2"]]  = 0.258
    zpe[gas["NH3"]] = 0.895
    """

    # coverage
    theta = np.zeros(len(ads))

    # entropy in eV/K
    deltaSs    = np.zeros(rxn_num)
    deltaSs[0] = -1.00e-3  # NH3 adsorption
    deltaSs[1] =  1.00e-3  # H2 desorption
    deltaSs[5] =  1.00e-3  # N3 desorption

    """
    # zero-point energy (ZPE) in eV
    deltaZPEs    = np.zeros(rxn_num)
    deltaZPEs[0] = -zpe[gas["NH3"]]  # NH3 adsorption
    deltaZPEs[1] =  zpe[gas["H2"]]    # H2 formation
    deltaZPEs[5] =  zpe[gas["N2"]]    # N2 formation
    """

    # thermal correction (translation + rotation) in eV
    deltaTherms    = np.zeros(rxn_num)
    deltaTherms[0] = -((3/2)*RT + RT)  # NH3 adsorption
    deltaTherms[1] =  ((3/2)*RT - RT)   # H2 desorption
    deltaTherms[5] =  ((3/2)*RT - RT)   # N2 desorption

    # pressure correction i.e. deltaGs = deltaGs^0 + RT*ln(p/p0)
    RTlnP = np.zeros(rxn_num)
    RTlnP[0] -= RT*np.log(p[gas["N2"]])
    RTlnP[1] -= RT*np.log(p[gas["H2"]])
    RTlnP[5] += RT*np.log(p[gas["NH3"]])

    # reation energies (in eV) and equilibrium constant
    deltaEs  = np.array(deltaEs)
    # deltaHs  = deltaEs + deltaZPEs + deltaTherms
    deltaHs  = deltaEs + deltaTherms
    deltaGs  = deltaHs - T*deltaSs
    deltaGs += RTlnP

    # Equilibrium constant
    K = np.exp(-deltaGs/RT)

    # activation energy
    # Bronsted-Evans-Polanyi --- universal a and b for stepped surface (Norskov 2002)
    alpha = 0.87
    beta = 1.34 + 1.0
    Ea = alpha*deltaEs[rds] + beta

    # A = 1.2e6 / np.sqrt(T)  # Dahl J.Catal., converted from bar^-1 to Pa^-1
    A = 0.241  # [s^-1] Logadottir, J.Catal 220 273 2003
    k = A*np.exp(-Ea/RT)

    # coverage

    # K[0] = thetaNH3/pNH3*theta -> thetaNH3 = K[0]*pNH3*theta
    # K[1] = pH2*theta^2/thetaH^2 -> thetaH = sqrt(pH2/K[1])*theta
    # K[2] = thetaNH2*thetaH / thetaNH3*theta -> thetaNH2 = K[2]*K[0]*pNH3*theta/sqrt(pH2/K[1])
    # K[3] = thetaNH*thetaH / thetaNH2*theta -> thetaNH = K[3]*K[2]*K[0]*pNH3*theta/(pH2/K[1])
    # K[4] = thetaN*thetaH / thetaNH*theta -> thetaN = K[4]*K[3]*K[2]*K[0]*pNH3*theta*/(pH2/K[1])^(3/2)

    denom = 1 + K[0]*p[gas["NH3"]] \
              + (p[gas["H2"]]/K[1])**(1/2) \
              + K[0]*K[2]*p[gas["NH3"]]/(p[gas["H2"]]/K[1])**(1/2) \
              + K[0]*K[2]*K[3]*p[gas["NH3"]]/(p[gas["H2"]]/K[1]) \
              + K[0]*K[2]*K[3]*K[4]*p[gas["NH3"]]/(p[gas["H2"]]/K[1])**(3/2)

    theta[ads["vac"]] = 1/denom
    theta[ads["NH3"]] = K[0]*p[gas["NH3"]]*theta[ads["vac"]]
    theta[ads["H"]]   = (p[gas["H2"]]/K[1])**(1/2)*theta[ads["vac"]]
    theta[ads["NH2"]] = K[0]*K[2]*p[gas["NH3"]]*theta[ads["vac"]] / (p[gas["H2"]]/K[1])**(1/2)
    theta[ads["NH"]]  = K[0]*K[2]*K[3]*p[gas["NH3"]]*theta[ads["vac"]] / (p[gas["H2"]]/K[1])
    theta[ads["N"]]   = K[0]*K[2]*K[3]*K[4]*p[gas["NH3"]]*theta[ads["vac"]] / (p[gas["H2"]]/K[1])**(3/2)

    Keq   = K[0]**1*K[1]**(3/2)*K[2]**1*K[3]**1*K[4]**1*K[5]**(1/2)
    gamma = (1/Keq)*(p[gas["N2"]]**(1/2)*p[gas["H2"]]**(3/2)/p[gas["NH3"]])
    rate  = k*p[gas["NH3"]]*theta[ads["vac"]]*(1-gamma)  # maybe TOF

    if debug:
        print("K[0]={0:4.2e}, K[1]={1:4.2e}, K[2]={2:4.2e}, K[3]={3:4.2e}, K[4]={4:4.2e}, K[5]={5:4.2e}".format(K[0],K[1],K[2],K[3],K[4],K[5]))
        print("theta[N]={0:4.2f}, theta[H]={1:4.2f}, theta[NH]={2:4.2f}, theta[NH2]={3:4.2f}, theta[NH3]={4:4.2f}, theta[vac]={5:4.2f}"
              .format(theta[ads["N"]], theta[ads["H"]], theta[ads["NH"]], theta[ads["NH2"]], theta[ads["NH3"]], theta[ads["vac"]]))
        print(f"Ea = {Ea:5.3f}, k = {k:5.3e}, Keq = {Keq:5.3e}, gamma = {gamma:5.3e}")
        print(f"rate = {rate:5.3e}")

    return rate
