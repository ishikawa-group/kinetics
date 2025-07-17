from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import numpy as np


def convert_numpy_types(obj):
    """Convert NumPy types to standard Python types"""
    import numpy as np

    if isinstance(obj, np.number):
        return obj.item()  # Convert NumPy numeric types to Python standard numeric types
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj


def get_number_of_layers(atoms):
    """
    Calculate the number of layers in a model based on atomic z-coordinates.
    
    Args:
        atoms: ASE atoms object
        
    Returns:
        int: Number of layers
    """
    import numpy as np

    positions = atoms.positions
    # Round z-coordinates for layer identification (3 decimal places)
    z_positions = np.round(positions[:, 2], decimals=3)
    num_layers = len(set(z_positions))
    return num_layers


def set_tags_by_z(atoms):
    """
    Set tags for each layer based on atomic z-coordinates.
    Each layer is assigned tags 0, 1, 2... from bottom to top.
    
    Args:
        atoms: ASE atoms object
        
    Returns:
        ASE atoms object with tags set
    """
    import numpy as np
    import pandas as pd

    new_atoms = atoms.copy()
    positions = new_atoms.positions
    # Round to 1 decimal place (used as layer width guideline)
    z_positions = np.round(positions[:, 2], decimals=1)

    # Extract unique layer values and ensure ascending order
    bins = np.sort(np.array(list(set(z_positions)))) + 1.0e-2
    bins = np.insert(bins, 0, 0)

    # Set labels for each interval (0, 1, 2, ...)
    labels = list(range(len(bins) - 1))
    tags = pd.cut(z_positions, bins=bins, labels=labels, include_lowest=True).tolist()
    new_atoms.set_tags(tags)

    return new_atoms


def fix_lower_surface(atoms):
    """
    Fix the bottom half layers of the model.
    First, set tags based on z-coordinates, then fix atoms in the bottom half layers.
    
    Example: For 3 layers, floor(3/2)=1, so the bottom 1 layer is fixed.
    
    Args:
        atoms: ASE atoms object
        
    Returns:
        ASE atoms object with bottom half fixed
    """
    import numpy as np
    from ase.constraints import FixAtoms

    atom_fix = atoms.copy()

    # Set tags (layer numbers from bottom)
    atom_fix = set_tags_by_z(atom_fix)
    tags = atom_fix.get_tags()

    # Get total number of layers
    num_layers = get_number_of_layers(atom_fix)
    # Bottom half layer numbers (rounded down)
    lower_layers = list(range(num_layers // 2))

    # Select atomic indices to fix
    fix_indices = [atom.index for atom in atom_fix if atom.tag in lower_layers]

    # Apply FixAtoms constraint
    constraint = FixAtoms(indices=fix_indices)
    atom_fix.set_constraint(constraint)

    return atom_fix


def parallel_displacement(atoms, vacuum=15.0):
    """
    Translate slab in z-direction so the lowest point becomes z=0,
    and add specified vacuum layer (vacuum[Å]) to the top (positive z-direction).
    
    Note:
        - This function assumes that the slab's surface normal direction aligns with the z-axis.
        - For oblique cells, perform rotation preprocessing beforehand.
    
    Args:
        atoms: ASE Atoms object (slab, preferably generated without vacuum option)
        vacuum: Thickness of vacuum layer to add (Å). Default is 15.0 Å.
    
    Returns:
        New ASE Atoms object with atomic positions shifted to z=0 bottom alignment
        and cell z-axis length set to (slab height + vacuum).
    """
    # Create copy to avoid modifying original object
    slab = atoms.copy()

    # Get current atomic positions and calculate minimum z value
    positions = slab.get_positions()
    z_min = positions[:, 2].min()

    # Translate entire slab in z-direction so lowest point becomes z=0
    slab.translate([0, 0, -z_min])

    # Get maximum z coordinate after translation
    z_max = slab.get_positions()[:, 2].max()
    # Calculate new z-axis length (slab height + vacuum)
    new_z_length = z_max + vacuum

    # Get cell matrix and set z-direction size to new length
    # Here we assume the cell's third vector aligns with z-direction
    cell = slab.get_cell().copy()
    # For safety, reset z-axis component to [0, 0, new_z_length]
    cell[2] = [0.0, 0.0, new_z_length]
    slab.set_cell(cell, scale_atoms=False)  # scale_atoms=False updates only cell, not atomic coordinates

    return slab


def auto_lmaxmix(atoms):
    """Automatically set lmaxmix when d/f elements are present"""
    d_elements = {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"
    }
    f_elements = {
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu", "Ac", "Th", "Pa", "U", "Np",
        "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"
    }

    symbols = set(atoms.get_chemical_symbols())

    if symbols & f_elements:
        lmaxmix_value = 6
    elif symbols & d_elements:
        lmaxmix_value = 4
    else:
        lmaxmix_value = 2

    atoms.calc.set(lmaxmix=lmaxmix_value)
    return atoms


def set_initial_magmoms(atoms, kind: str = "bulk", formula: str = None):
    """
    Set initial magnetic moments for atoms
    
    Args:
        atoms: ASE atoms object
        kind: "gas" / "slab" / "bulk" - system type
        formula: Molecular formula (used when kind is "gas")
        
    Returns:
        atoms: Atoms object with magnetic moments set
    """
    # Define constants within function
    MAGNETIC_ELEMENTS = ["Mn", "Fe", "Cr", "Ni"]  # Initial magnetic moment 1.0 μB
    CLOSED_SHELL_MOLECULES = ["H2", "H2O"]  # Molecules calculated with spin unpolarized

    symbols = atoms.get_chemical_symbols()

    # For gas phase closed-shell molecules, set all to 0
    if kind == "gas" and formula in CLOSED_SHELL_MOLECULES:
        initial_magmom = [0.0] * len(symbols)
    else:
        # Set 1.0 μB for magnetic elements, 0.0 for others
        initial_magmom = [1.0 if symbol in MAGNETIC_ELEMENTS else 0.0 for symbol in symbols]

    atoms.set_initial_magnetic_moments(initial_magmom)
    return atoms


def sort_atoms(atoms, axes=("z", "y", "x")):
    """
    Sort Atoms object by specified axis order (default is (z, y, x)).
    
    Parameters:
        atoms (ase.Atoms): Atomic structure to sort
        axes (tuple): Axes for sorting. Example: ("z", "y", "x")
        
    Returns:
        sorted_atoms (ase.Atoms): Atoms object sorted by specified axis order
    """
    import numpy as np

    axis_map = {"x": 0, "y": 1, "z": 2}
    positions = atoms.get_positions()  # shape: (n_atoms, 3)

    # lexsort: last given key has highest priority, so pass axes[::-1]
    keys = tuple(positions[:, axis_map[ax]] for ax in axes[::-1])
    sorted_indices = np.lexsort(keys)

    sorted_atoms = atoms[sorted_indices]
    sorted_atoms.set_tags(atoms.get_tags())
    sorted_atoms.set_cell(atoms.get_cell())
    sorted_atoms.set_pbc(atoms.get_pbc())

    return sorted_atoms


def slab_to_tensor(slab, grid_size):
    """
    Convert slab structure to 3D tensor based on grid size [x, y, z].
    
    First sorts by (z, y, x) axes to reshape into (z, y, x) format,
    then inserts 0s alternately in x and y directions.
    For each z layer, even layers (z index even) use basic pattern (even rows, even columns),
    odd layers (z index odd) use shifted pattern (odd rows, odd columns).
    
    Parameters:
        slab (ase.Atoms): Slab structure to convert (number of atoms must match x*y*z)
        grid_size (list or tuple): [x, y, z] (example: [8, 8, 3])
        
    Returns:
        tensor (torch.Tensor): Interleaved 3D tensor
        Final shape is (z, new_y, new_x) where new_y = 2*y, new_x = 2*x
    """
    import torch

    x_size, y_size, z_size = grid_size  # grid_size = [x, y, z]
    total_cells = x_size * y_size * z_size

    if len(slab) != total_cells:
        raise ValueError(f"Number of atoms in slab {len(slab)} does not match grid cells {total_cells}")

    # Sort and reshape to (z, y, x) format
    sorted_slab = sort_atoms(slab, axes=("z", "y", "x"))
    basic_tensor = torch.tensor(
        sorted_slab.get_atomic_numbers(),
        dtype=torch.int64
    ).reshape(z_size, y_size, x_size)

    # New tensor size (x,y directions: 2x)
    new_x_size = 2 * x_size
    new_y_size = 2 * y_size

    # z remains the same
    interleaved = torch.zeros((z_size, new_y_size, new_x_size), dtype=torch.int64)

    # Set pattern for each z layer
    for z in range(z_size):
        if z % 2 == 0:
            # Even z layer (human 1st, 3rd layer...):
            # Basic tensor row i goes to interleaved row 2*i,
            # columns also at even indices (0,2,4,...)
            interleaved[z, 0::2, 0::2] = basic_tensor[z, :, :]
        else:
            # Odd z layer (human 2nd, 4th layer...):
            # Basic tensor row i goes to interleaved row 2*i+1,
            # columns also at odd indices (1,3,5,...)
            interleaved[z, 1::2, 1::2] = basic_tensor[z, :, :]

    return interleaved


def tensor_to_slab(tensor, template_slab):
    """
    Restore slab structure (ASE Atoms object) from interleaved tensor.
    For slab_to_tensor case, for each z layer:
      - If z layer is even, interleaved[z, 0::2, 0::2] elements are original atomic numbers
      - If z layer is odd, interleaved[z, 1::2, 1::2] elements are original atomic numbers
    Extract each and reshape to original order ((z, y, x)).
    
    Parameters:
        tensor (torch.Tensor): 3D tensor, shape is (z, new_y, new_x) with new_y = 2*y, new_x = 2*x
        template_slab (ase.Atoms): Template for restoration (original slab structure, sorted)
        
    Returns:
        new_slab (ase.Atoms): Slab structure restored with tensor information
    """
    import torch

    z_size, new_y_size, new_x_size = tensor.shape

    # Restore original y, x sizes (new size is 2x)
    y_size = new_y_size // 2
    x_size = new_x_size // 2
    total_atoms = z_size * y_size * x_size

    if total_atoms != len(template_slab):
        raise ValueError("Number of atoms to restore from tensor does not match template_slab")

    # Create list for each z layer
    reconstructed = []
    for z in range(z_size):
        if z % 2 == 0:
            # Even z layer: extract interleaved[z, 0::2, 0::2]
            layer = tensor[z, 0::2, 0::2]
        else:
            # Odd z layer: extract interleaved[z, 1::2, 1::2]
            layer = tensor[z, 1::2, 1::2]
        # layer has shape (y_size, x_size)
        reconstructed.append(layer.flatten())

    # Concatenate to (z*y_size*x_size,) 1D array
    new_atomic_nums = torch.cat(reconstructed).numpy()

    new_slab = template_slab.copy()
    new_slab.set_atomic_numbers(new_atomic_nums)

    return new_slab


def generate_result_csv(
        materials_data: Dict[str, str],
        output_csv: str = "orr_results.csv",
        verbose: bool = False,
    ) -> Optional[str]:
    """
    Compile ORR calculation results for multiple materials into CSV file
    
    Args:
        materials_data: Dictionary of material names and all_results.json paths 
                       {'Pt111': 'path/to/Pt111/all_results.json', ...}
        output_csv: Output CSV file path
        verbose: Whether to show detailed output
        
    Returns:
        Path of generated CSV file
    """
    import json
    import csv
    from pathlib import Path
    from orr_overpotential_calculator.calc_orr_overpotential import compute_reaction_energies, \
        get_overpotential_orr

    # Data for CSV output
    csv_data = []

    # Process data for each material
    for material_name, json_path in materials_data.items():
        if verbose:
            print(f"Processing {material_name}...")

        # Load JSON data
        try:
            with open(json_path, 'r') as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue

        # Get slab energy
        E_slab = results["OH"]["E_slab"]

        # Calculate reaction energies
        try:
            deltaEs, energies = compute_reaction_energies(results, E_slab)

            # Calculate overpotential (set output_dir to None to avoid file output if needed)
            output_dir = Path(json_path).parent if verbose else None
            orr_results = get_overpotential_orr(deltaEs, output_dir, verbose=verbose, save_plot=False)

            # Extract values from dictionary
            eta = orr_results["eta"]
            diffG_U0 = orr_results["diffG_U0"]
            diffG_eq = orr_results["diffG_eq"]
            U_L = orr_results["U_L"]

            # Extract adsorption energies
            E_ads_OOH = results.get("HO2", {}).get("E_ads_best", None)
            E_ads_O = results.get("O", {}).get("E_ads_best", None)
            E_ads_OH = results.get("OH", {}).get("E_ads_best", None)

            # Create row data
            row_data = {
                "Material": material_name,
                "E_slab": E_slab,
                "E_H2_g": energies["E_H2_g"],
                "E_H2O_g": energies["E_H2O_g"],
                "E_O2_g": energies["E_O2_g"],
                "E_slab_OOH": energies["E_slab_OOH"],
                "E_slab_O": energies["E_slab_O"],
                "E_slab_OH": energies["E_slab_OH"],
                "E_ads_OOH": E_ads_OOH,
                "E_ads_O": E_ads_O,
                "E_ads_OH": E_ads_OH,
                "dG1": diffG_U0[0],
                "dG2": diffG_U0[1],
                "dG3": diffG_U0[2],
                "dG4": diffG_U0[3],
                "dG_eq_1": diffG_eq[0],
                "dG_eq_2": diffG_eq[1],
                "dG_eq_3": diffG_eq[2],
                "dG_eq_4": diffG_eq[3],
                "U_L": U_L,
                "Overpotential": eta,
                "Limiting potential": 1.23 - eta,
            }

            csv_data.append(row_data)

            if verbose:
                print(f"  {material_name}: η = {eta:.3f} V")

        except Exception as e:
            print(f"Error processing {material_name}: {e}")

    # Write to CSV file
    if not csv_data:
        print("No data to write to CSV!")
        return None

    # Set headers (include all data columns)
    fieldnames = list(csv_data[0].keys())

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"CSV file generated: {output_csv}")
    return output_csv


def create_orr_volcano_plot(
        csv_file: Union[str, Path],
        output_file: str = "orr_volcano.png",
        x_column: str = "dG_OH",
        y_column: str = "Limiting potential",
        label_column: str = "Material",
        dpi: int = 300,
        figsize: tuple = (10, 10),
        markersize: int = 80,
        ideal_line: float = 1.23,
    ) -> str:
    """
    Generate ORR volcano plot (dG_OH vs Limiting potential)
    
    Args:
        csv_file: Input CSV file path
        output_file: Output image filename
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        label_column: Column name for legend
        dpi: Image resolution
        figsize: Figure size (width, height) in inches
        markersize: Marker size
        ideal_line: Ideal limiting potential value (V)
        
    Returns:
        Path of saved image
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    # Load CSV file
    df = pd.read_csv(csv_file)

    # -------------- Calculate dG_OH -----------------
    # Constants
    T = 298.15  # K

    # ZPE (eV)
    zpe = {
        "H2": 0.35, "H2O": 0.57,
        "Oads": 0.06, "OHads": 0.37, "OOHads": 0.44,
    }

    # Entropy term TS (eV) at 298 K
    TS_H2 = 0.403  # eV
    TS_H2O = 0.67  # eV
    TS_OHads = 0.0  # eV

    # Electronic part ΔE_OH
    # Reaction: H2O + * -> OH* + 1/2 H2
    df["E_slab_OH"] = df["E_slab_OH"] - 0.1  # solvent correction
    df["dE_OH"] = df["E_slab_OH"] - df["E_slab"] - (df["E_H2O_g"] - 0.5 * df["E_H2_g"])

    # ZPE difference
    delta_zpe = zpe["OHads"] - (zpe["H2O"] - 0.5 * zpe["H2"])  # eV

    # -TΔS term (products - reactants) (adsorbate entropies ≈ 0)
    delta_TS = (0.5 * TS_H2 + TS_OHads) - TS_H2O  # eV

    # ΔG_OH
    df["dG_OH"] = df["dE_OH"] + delta_zpe - delta_TS

    # Set font size
    plt.rcParams.update({'font.size': 12})

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set colormap
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, len(df) - 1)

    # Plot each material with explicitly defined colors
    colors = [cmap(norm(i)) for i in range(len(df))]

    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(
            row[x_column],
            row[y_column],
            color=colors[i],
            s=markersize,
            alpha=0.8,
            edgecolor='black',
            label=row[label_column]
        )

    # Add labels to each point
    for i, (_, row) in enumerate(df.iterrows()):
        ax.annotate(
            row[label_column],
            (row[x_column], row[y_column]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold'
        )

    # Set x and y axis ranges
    ax.set_xlim(-0.25, 1.75)
    ax.set_ylim(-0.5, 1.5)

    # Add theoretical lines
    # Ideal limiting potential (1.23V) horizontal line
    ax.axhline(y=ideal_line, color='k', linestyle="solid", linewidth=1.5, alpha=0.7,
               label=f'Ideal ({ideal_line} V)')

    # Additional theoretical lines
    x_vals = np.linspace(-1, 3, 100)

    # OH* -> H2O: y = -x + 1.72
    y_vals_1 = -x_vals + 1.72
    ax.plot(x_vals, y_vals_1, color='k', linestyle="dotted", alpha=0.7, linewidth=1.5, label='OH* -> H2O')

    # O2 -> HOO*: y = x
    y_vals_2 = x_vals
    ax.plot(x_vals, y_vals_2, color='k', linestyle="dashed", alpha=0.7, linewidth=1.5, label='O2 -> HOO*')

    # x = 0.86 vertical line
    # ax.axvline(x=0.86, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='x = 0.86')

    # Graph settings
    ax.set_xlabel(f'ΔG_OH (eV)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{y_column} (V)', fontsize=14, fontweight='bold')
    ax.set_title('ORR Volcano Plot', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Material legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                          markersize=10, markeredgecolor='black') for i in range(len(df))]
    legend1 = ax.legend(handles, df[label_column],
                        title='Materials',
                        loc='upper right',
                        frameon=True)
    ax.add_artist(legend1)

    # Theoretical lines legend
    handles2, labels2 = [], []
    for line in ax.lines:
        if line.get_label() and not line.get_label().startswith('_'):
            handles2.append(line)
            labels2.append(line.get_label())

    legend2 = ax.legend(handles2, labels2,
                        loc='lower right',
                        frameon=True)

    # Adjust graph layout
    plt.tight_layout()

    # Save image
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Volcano plot saved: {output_path}")
    return str(output_path)


def place_adsorbate(cluster, adsorbate, indices, height=None, orientation=None):
    """
    Place adsorbate molecule on cluster surface at specified binding sites.
    
    Args:
        cluster: ASE Atoms object 
        adsorbate: ASE Atoms object representing the adsorbate molecule
        indices: List of atom indices defining the binding site
        height: Distance between adsorbate and surface (default: 2.0 Å)
        orientation: Custom orientation vector for adsorbate placement
        
    Returns:
        ASE Atoms object with adsorbate placed 
    """
    import numpy as np
    from ase import Atoms

    cluster_center = cluster.get_center_of_mass()
    indices = list(indices)

    if len(indices) == 1:  # -------- on-top
        base_atom = indices[0]
        pos = cluster.positions[base_atom]
        normal = pos - cluster_center

    elif len(indices) == 2:  # -------- bridge
        i1, i2 = indices
        p1, p2 = cluster.positions[[i1, i2]]
        pos = (p1 + p2) / 2.0
        edge_vec = p2 - p1
        center_vec = pos - cluster_center

        # ① perpendicular to edge_vec, ② keep only outward component
        normal = np.cross(edge_vec, np.cross(center_vec, edge_vec))
        if np.linalg.norm(normal) < 1e-6:
            normal = center_vec.copy()  # Use radial when degenerate

        # Align outward
        if np.dot(normal, center_vec) < 0:
            normal = -normal

    elif len(indices) == 3:  # -------- 3-fold hollow
        i1, i2, i3 = indices
        p1, p2, p3 = cluster.positions[[i1, i2, i3]]
        pos = (p1 + p2 + p3) / 3.0
        normal = np.cross(p2 - p1, p3 - p1)
        if np.dot(normal, pos - cluster_center) < 0:
            normal = -normal

    elif len(indices) == 4:  # -------- 4-fold hollow
        i1, i2, i3, i4 = indices
        p1, p2, p3, p4 = cluster.positions[[i1, i2, i3, i4]]
        pos = (p1 + p2 + p3 + p4) / 4.0
        normal = np.cross(p2 - p1, p3 - p1)
        if np.dot(normal, pos - cluster_center) < 0:
            normal = -normal

    else:
        raise ValueError("indices should contain 1-4 elements")

    # ------------ Rest follows original logic ------------
    if orientation is not None:
        orientation = np.asarray(orientation, float)
        if np.linalg.norm(orientation) < 1e-6:
            raise ValueError("orientation vector has zero length")
        normal = orientation
    normal /= np.linalg.norm(normal)

    if height is None:
        height = 2.0

    ads = adsorbate.copy()
    if len(ads) == 0:
        raise ValueError("adsorbate is empty")

    anchor_pos = ads.positions[0].copy()
    ads.positions -= anchor_pos  # Move anchor atom to origin

    if len(ads) > 1:
        v_ads = ads.positions[1]  # v_ads = origin→second atom
        ads.rotate(v_ads, normal, center=(0, 0, 0))  # Align 1st-2nd atom axis to normal

    ads.translate(pos + normal * height)  # Offset by height

    combined = cluster.copy()
    combined += ads
    return combined


def plot_free_energy_diagram(
        csv_file: Union[str, Path],
        output_file: str = "free_energy_diagram.png",
        equilibrium_potential: float = 1.23,
        dpi: int = 300,
        figsize: tuple = (10, 8),
        show_u0: bool = True,
        show_ueq: bool = True,
        material_name: Optional[str] = None,
    ) -> str:
    """
    Generate combined free energy diagram for multiple materials from CSV data.
    Displays energy levels as horizontal lines connected by dashed lines.
    
    Args:
        csv_file: Input CSV file path containing ORR calculation results
        output_file: Output image filename
        equilibrium_potential: Equilibrium potential in V (default: 1.23 V)
        dpi: Image resolution
        figsize: Figure size (width, height) in inches
        show_u0: Whether to show U=0V profiles
        show_ueq: Whether to show U=1.23V profiles
        material_name: Optional specific material name to plot (from CSV)
        
    Returns:
        Path of saved image
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    # Load CSV data
    df = pd.read_csv(csv_file)

    # Filter by material_name if specified
    if material_name is not None:
        filtered_df = df[df["Material"] == material_name]
        if len(filtered_df) == 0:
            raise ValueError(f"Material '{material_name}' not found in CSV file")
        df = filtered_df

    # Reaction step labels
    labels = [
        "O$_2$ + 2H$_2$",
        "OOH* + 1.5H$_2$",
        "O* + H$_2$O + H$_2$",
        "OH* + H$_2$O + 0.5H$_2$",
        "* + 2H$_2$O"
    ]
    steps = np.arange(5)  # 0, 1, 2, 3, 4

    # Color palette for materials
    if material_name is not None:
        # Use fixed colors for single material mode
        u0_color = 'black'  # Black
        ul_color = 'blue'  # Blue
        ueq_color = 'green'  # Green
        colors = [None]  # Dummy (not used)
    else:
        # Use different colors for each material in multi-material mode
        colors = [plt.cm.tab10(i) for i in range(len(df))]

    # Create figure
    plt.figure(figsize=figsize)

    # Horizontal line width (extension from reaction coordinate)
    line_width = 0.3

    # Process each material
    for idx, (_, row) in enumerate(df.iterrows()):
        material_name_row = row["Material"]
        limiting_potential = row["Limiting potential"]

        # Extract pre-calculated dG values from CSV
        dg_u0 = np.array([row["dG1"], row["dG2"], row["dG3"], row["dG4"]])
        dg_eq = np.array([row["dG_eq_1"], row["dG_eq_2"], row["dG_eq_3"], row["dG_eq_4"]])

        # Calculate dG_UL using U_L (Limiting potential)
        dg_ul = dg_u0 + limiting_potential

        # Free energy profiles (cumulative sum starting from 0)
        g_profile_u0 = np.concatenate(([0.0], np.cumsum(dg_u0)))
        g_profile_ueq = np.concatenate(([0.0], np.cumsum(dg_eq)))
        g_profile_ul = np.concatenate(([0.0], np.cumsum(dg_ul)))

        # Shift profiles so final state is at 0
        g0_shift = g_profile_u0 - g_profile_u0[-1]
        geq_shift = g_profile_ueq - g_profile_ueq[-1]
        gul_shift = g_profile_ul - g_profile_ul[-1]

        # Color selection for each material
        if material_name is not None:
            # Single material mode: fixed colors
            color_u0 = u0_color
            color_ul = ul_color
            color_ueq = ueq_color
        else:
            # Multi-material mode: same color for each material
            color_u0 = colors[idx]
            color_ul = colors[idx]
            color_ueq = colors[idx]

        # Labels for legend
        if material_name is not None:
            u0_label = "U = 0V" if show_u0 else None
            ueq_label = f"U = {equilibrium_potential}V" if show_ueq else None
            ul_label = f"U$_{{L}}$ = {limiting_potential:.2f}V"
        else:
            u0_label = f"{material_name_row} (U=0V)" if show_u0 else None
            ueq_label = f"{material_name_row} (U={equilibrium_potential}V)" if show_ueq else None
            ul_label = f"{material_name_row} (U$_{{L}}$={limiting_potential:.2f}V)"

        # ------ U=0V profile ------
        if show_u0:
            # Show label only for first point
            plt.hlines(g0_shift[0], steps[0] - line_width, steps[0] + line_width,
                       color=color_u0, alpha=0.6, linewidth=2.5, label=u0_label)

            # Horizontal lines for remaining points (no label)
            for i in range(1, len(steps)):
                plt.hlines(g0_shift[i], steps[i] - line_width, steps[i] + line_width,
                           color=color_u0, alpha=0.6, linewidth=2.5)

            # Dashed line connections between consecutive points
            for i in range(len(steps) - 1):
                plt.plot([steps[i] + line_width, steps[i + 1] - line_width],
                         [g0_shift[i], g0_shift[i + 1]],
                         '--', color=color_u0, alpha=0.6, linewidth=1.0)

            # Add markers
            plt.plot(steps, g0_shift, 'o', color=color_u0, alpha=0.6,
                     markersize=4, linestyle='none')

        # ------ U=U_L profile ------
        # Show label only for first point
        plt.hlines(gul_shift[0], steps[0] - line_width, steps[0] + line_width,
                   color=color_ul, linewidth=2.5, label=ul_label)

        # Horizontal lines for remaining points (no label)
        for i in range(1, len(steps)):
            plt.hlines(gul_shift[i], steps[i] - line_width, steps[i] + line_width,
                       color=color_ul, linewidth=2.5)

        # Dashed line connections between consecutive points
        for i in range(len(steps) - 1):
            plt.plot([steps[i] + line_width, steps[i + 1] - line_width],
                     [gul_shift[i], gul_shift[i + 1]],
                     '--', color=color_ul, linewidth=1.0)

        # Add markers
        plt.plot(steps, gul_shift, 's', color=color_ul, markersize=5, linestyle='none')

        # ------ U=1.23V profile ------
        if show_ueq:
            # Show label only for first point
            plt.hlines(geq_shift[0], steps[0] - line_width, steps[0] + line_width,
                       color=color_ueq, alpha=0.8, linewidth=2.5, label=ueq_label)

            # Horizontal lines for remaining points (no label)
            for i in range(1, len(steps)):
                plt.hlines(geq_shift[i], steps[i] - line_width, steps[i] + line_width,
                           color=color_ueq, alpha=0.8, linewidth=2.5)

            # Dashed line connections between consecutive points
            for i in range(len(steps) - 1):
                plt.plot([steps[i] + line_width, steps[i + 1] - line_width],
                         [geq_shift[i], geq_shift[i + 1]],
                         '--', color=color_ueq, alpha=0.8, linewidth=1.0)

            # Add markers
            plt.plot(steps, geq_shift, 'o', color=color_ueq, alpha=0.8,
                     markersize=6, linestyle='none')

    # Formatting
    plt.xticks(steps, labels, rotation=15, ha='right')
    plt.ylabel("ΔG (eV)", fontsize=12, fontweight='bold')
    plt.xlabel("Reaction Coordinate", fontsize=12, fontweight='bold')

    # Title setting
    if material_name is not None:
        plt.title(f"{material_name} - ORR Free Energy Diagram",
                  fontsize=14, fontweight='bold')
    else:
        plt.title("4e⁻ ORR Free Energy Diagrams - Material Comparison",
                  fontsize=14, fontweight='bold')

    # Grid and legend
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)

    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Free energy diagram saved: {output_path}")
    return str(output_path)
