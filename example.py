import autode as ade
import os
import numpy as np
import matplotlib.pyplot as plt
import pyscf
from utils.shannon import calc_coper
from pyscf import gto

def read_xyz(filename):
    """Read XYZ file and return coordinates and atoms."""
    with open(filename, 'r') as f:
        lines = f.readlines()
        structures = []
        atoms_list = []
        i = 0
        while i < len(lines):
            if len(lines[i].strip()) > 0:  # Check if line is not empty
                n_atoms = int(lines[i])
                coords = []
                atoms = []
                for j in range(n_atoms):
                    if i + 2 + j >= len(lines):
                        break
                    line = lines[i + 2 + j]
                    if len(line.split()) == 4:
                        atom, x, y, z = line.split()
                        coords.append([float(x), float(y), float(z)])
                        atoms.append(atom)
                if coords:
                    structures.append(np.array(coords))
                    atoms_list.append(atoms)
                i += n_atoms + 2
            else:
                i += 1
    return (structures[0], atoms_list[0]) if len(structures) == 1 else (structures, atoms_list)

def interpolate_structures(start, end, n_points):
    """Linear interpolation between two structures."""
    structures = []
    for i in range(n_points):
        t = i / (n_points - 1)
        interpolated = start + t * (end - start)
        structures.append(interpolated)
    return structures

def coords_to_pyscf(coords, atoms, basis):
    """Convert coordinates and atoms directly to PySCF molecule."""
    mol = gto.Mole()
    mol.atom = [[atom, coord] for atom, coord in zip(atoms, coords)]
    mol.basis = basis
    # Set spin to 1 (doublet) since we have a radical with odd number of electrons
    mol.spin = 1
    mol.build()
    return mol

def calculate_reaction_profile(reactant_file, ts_file, product_file, n_interpolation_points=10, ts_stride=10, basis="6-31G*", method="b3lyp"):
    """Calculate reaction profile with configurable number of points"""
    # Read structures
    reactant_coords, reactant_atoms = read_xyz(reactant_file)
    ts_frames_coords, ts_frames_atoms = read_xyz(ts_file)
    product_coords, product_atoms = read_xyz(product_file)

    # Ensure all structures have same number of atoms
    if len(ts_frames_coords[0]) != len(reactant_coords):
        raise ValueError(f"Inconsistent number of atoms: TS={len(ts_frames_coords[0])}, reactant={len(reactant_coords)}")

    # Calculate Shannon entropy for each structure
    shannon_values = []
    reaction_coordinate = []

    # Process reactant -> first TS frame
    structures_r_to_ts = interpolate_structures(reactant_coords, ts_frames_coords[0], n_interpolation_points)
    for i, coords in enumerate(structures_r_to_ts):
        mol = coords_to_pyscf(coords, reactant_atoms, basis)
        shannon = calc_coper(mol, basis, method)
        shannon_values.append(shannon)
        reaction_coordinate.append(i / (n_interpolation_points - 1) * 33)  # Scale to 0-33

    # Process TS frames
    selected_ts_frames = ts_frames_coords[::ts_stride]
    for i, coords in enumerate(selected_ts_frames):
        mol = coords_to_pyscf(coords, ts_frames_atoms[0], basis)
        shannon = calc_coper(mol, basis, method)
        shannon_values.append(shannon)
        reaction_coordinate.append(33 + (i / len(selected_ts_frames)) * 34)  # Scale 33-67

    # Process last TS frame -> product
    structures_ts_to_p = interpolate_structures(ts_frames_coords[-1], product_coords, n_interpolation_points)
    for i, coords in enumerate(structures_ts_to_p[1:]):  # Skip first point (duplicate of last TS)
        mol = coords_to_pyscf(coords, product_atoms, basis)
        shannon = calc_coper(mol, basis, method)
        shannon_values.append(shannon)
        reaction_coordinate.append(67 + ((i + 1) / (n_interpolation_points - 1)) * 33)  # Scale 67-100

    return reaction_coordinate, shannon_values

# Calculate reaction profile
reaction_coordinate, shannon_values = calculate_reaction_profile(
    "cope_rearrangement/output/reactant.xyz",
    "cope_rearrangement/output/TS_imag_mode.xyz", 
    "cope_rearrangement/output/product.xyz"
)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(reaction_coordinate, shannon_values, 'b-', linewidth=2)
plt.scatter(reaction_coordinate, shannon_values, c='b', s=50)
plt.axvspan(33, 67, color='r', alpha=0.1, label='Transition State Region')
plt.xlabel('Reaction Coordinate (%)')
plt.ylabel('Shannon Information Content')
plt.title('Information Content Along Reaction Path')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('cope_rearrangement/reaction_profile.png')
plt.close()

# Print some diagnostic information
print(f"Total number of points calculated: {len(shannon_values)}")
print(f"Reaction coordinate range: {min(reaction_coordinate):.1f} to {max(reaction_coordinate):.1f}")