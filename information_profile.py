import numpy as np
from utils.shannon import calc_coper
from pyscf import gto
import os
import argparse

def read_xyz_frames(filename):
    """Read multiple XYZ frames from a file."""
    frames = []
    with open(filename, 'r') as f:
        while True:
            try:
                n_atoms = int(f.readline())
                f.readline()
                atoms = []
                for _ in range(n_atoms):
                    line = f.readline().strip().split()
                    atom = [line[0], float(line[1]), float(line[2]), float(line[3])]
                    atoms.append(atom)
                frames.append(atoms)
            except (ValueError, IndexError):
                break
    return frames

def frame_to_pyscf(frame, basis, spin=0, charge=0):
    """Convert a frame to PySCF molecule object."""
    mol = gto.Mole()
    mol.atom = frame
    mol.basis = basis
    mol.spin = spin  # Set spin to 1 for radical species (doublet state)
    mol.charge = charge
    mol.build()
    return mol

def process_frame(frame, basis, xc, l):
    """Process a single frame, returning both entropy and energy."""
    mol = frame_to_pyscf(frame, basis)
    entropy, energy = calc_coper(mol, basis, xc, l)

    return entropy, energy

def compute_information_profile(dir, basis, xc, l):
    # Read all frames
    xyz_file = os.path.join(dir, "irc/irc_IRC_Full_trj.xyz")
    frames = read_xyz_frames(xyz_file)

    # Process frames sequentially
    results = []
    for frame in frames:
        results.append(process_frame(frame, basis, xc, l))
    
    # Unzip results
    entropies, energies = zip(*results)
    entropies = np.array(entropies)
    energies = np.array(energies)

    # Save results
    results = np.column_stack((entropies, energies))
    np.savetxt(os.path.join(dir, f"profile_{basis}_{xc}_{l}.txt"), results, 
               header='CoPer Energy_Hartree', 
               fmt='%.6f')

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process IRC frames.')
    parser.add_argument('--dir', help='Directory containing IRC data')
    parser.add_argument('--basis', help='Basis set')
    parser.add_argument('--xc', help='Exchange-correlation functional')
    parser.add_argument('--l', help='Grid resolution')
    args = parser.parse_args()
    
   # Setup calculation parameters
    dir = args.dir
    basis = args.basis
    xc = args.xc
    l = int(args.l)

    compute_information_profile(dir, basis, xc, l)