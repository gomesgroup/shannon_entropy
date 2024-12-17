import autode as ade
import os

ade.Config.n_cores = 16

r = ade.Reactant(name='reactant', smiles='CC[C]([H])[H]')
p = ade.Product(name='product', smiles='C[C]([H])C')

reaction = ade.Reaction(r, p, name='1-2_shift')
reaction.calculate_reaction_profile()  # creates 1-2_shift/ and saves profile

# Create a directory for xyz files if it doesn't exist
xyz_dir = '1-2_shift/xyz_files'
if not os.path.exists(xyz_dir):
    os.makedirs(xyz_dir)

# Save xyz files for each point along the reaction path
if hasattr(reaction, 'profile') and reaction.profile is not None:
    for i, point in enumerate(reaction.profile.species):
        xyz_filename = os.path.join(xyz_dir, f'structure_{i:03d}.xyz')
        point.save_xyz_file(xyz_filename)