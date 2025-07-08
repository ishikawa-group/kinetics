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

#### Introduction
* Here we consider the OER consists of following four elementary reactions.
$$
\begin{align*}
\rm{H_2O + *}  &\rightarrow \rm{OH* + 1/2H_2}  \\
\rm{OH*}       &\rightarrow \rm{O* + 1/2H_2}   \\
\rm{O* + H_2O} &\rightarrow \rm{OOH* + 1/2H_2} \\
\rm{OOH*}      &\rightarrow \rm{O_2 + * + 1/2H_2}
\end{align*}
$$
* These reactions have following meanings:
  1. Water adsorption and dissociation
  2. OH dehydrogenation to O
  3. Water oxidation to form OOH
  4. O2 evolution
* To calculate the proton and electron Gibbs energy, the computational hydrogen electrode (CHE) is assumed:
$$
G_{\rm H^+} + G_{\rm e^-} = \frac{1}{2}G_{\rm H_2}
$$
* The Gibbs energy change of these reactions ($\Delta G$) is calcualted.
* The overpotential $\eta$ is calculated by taking the maximum $\Delta G$ of these, and subtracting equilibrium potential.
$$
\eta = \max\{\Delta G_1, \Delta G_2, \Delta G_3, \Delta G_4\} - U_0
$$

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
  - `formula`: chemical formula of the surface
  - `cell_volume`: bulk unit cell volume
  - `s_electrons`, `p_electrons`, `d_electrons`, `f_electrons`: number of s-, p-, d-, and f-electrons
  - `min_M_O_distance`: minimum metal-oxygen distance
  - `overpotential_in_eV`: calculated OER overpotential

#### Required Files
* CIF files (single or multiple).
* `oer.txt`: OER reaction pathway definition
   ```
   H2O + surf     --ads--> OH_fcc + 0.5*H2
   OH_fcc         --LH-->  O_fcc + 0.5*H2
   O_fcc + H2O    --LH-->  OOH_fcc + 0.5*H2
   OOH_fcc        --LH-->  O2 + surf + 0.5*H2
   ```


### 2. Oxygen reduction reaction (ORR)
* Similar to the OER, the library provides function to calculate the overpotential of ORR.
* Example code is `examples/orr/orr.py`.

#### Introduction
* Following elementary reactions are assumed.

$$
\begin{align*}
\rm{O_2 + H^+ + e^-}  &\rightarrow  \rm{OOH*} \\
\rm{OOH* + H^+ + e^-}  &\rightarrow \rm{O* + H_2O} \\
\rm{O* + H^+ + e^-}  &\rightarrow \rm{OH*} \\
\rm{OH* + H^+ + e^-}  &\rightarrow \rm{H_2O}
\end{align*}
$$
* Likewise the OER, CHE is assumed to calculate the proton and electron Gibbs energy.

#### Usage
```python
from kinetics.microkinetics.orr_and_oer import get_overpotential_oer_orr

# Reaction energies for each step (deltaEs)
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
