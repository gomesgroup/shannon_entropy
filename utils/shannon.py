import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pyscf
import openbabel as ob
from pyscf import gto, dft
from pyscf.tools import cubegen
import jax.numpy as jnp
from jax import jit
import struct

def calc_coper(input, basis, xc, l=100):
    mol = convert_pyscf(input, basis)

    mf = dft.RKS(mol)
    mf.xc = xc
    energy = mf.kernel()

    density = mf.make_rdm1()
    cube = cubegen.density(mol, "out.cube", density, nx=l, ny=l, nz=l)

    z = jnp.sum(cube)    # partition function
    cube = cube / z      # normalize densities
    entropy = -jnp.sum(cube * jnp.log(cube))   

    coper = jnp.exp(entropy - (3 * jnp.log(l)))

    return coper, energy

def calc_coper_fast(input, basis, xc, l=100, outfile="out.cube"):
    mol = convert_pyscf(input, basis)
    
    # Add parameters to speed up SCF convergence
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-6  # Slightly looser convergence tolerance (default is 1e-7)
    mf.max_cycle = 50   # Reduce maximum SCF cycles if convergence is usually quick
    mf.direct_scf = True  # Use direct SCF to save memory
    
    # Add initial guess to potentially speed up SCF convergence
    dm = mf.init_guess_by_atom()
    energy = mf.kernel(dm)

    density = mf.make_rdm1()
    # Add more efficient cube generation parameters
    cube = cubegen.density(mol, outfile, density, nx=l, ny=l, nz=l, 
                          resolution=None, margin=2.0)

    z = jnp.sum(cube)    # partition function
    cube = cube / z      # normalize densities
    entropy = -jnp.sum(cube * jnp.log(cube))   

    coper = jnp.exp(entropy - (3 * jnp.log(l)))

    return coper, energy

def calc_entropy(input, basis, xc, l):
    mol = convert_pyscf(input, basis)

    mf = dft.RKS(mol)
    mf.xc = xc
    energy = mf.kernel()

    density = mf.make_rdm1()
    cube = cubegen.density(mol, "out.cube", density, nx=l, ny=l, nz=l)

    z = jnp.sum(cube)    # partition function
    cube = cube / z      # normalize densities
    entropy = -jnp.sum(cube * jnp.log(cube))   

    return entropy

def convert_pyscf(input, basis):
    if type(input) is pyscf.gto.mole.Mole:
        pyscf_mol = input
    elif type(input) is ob.openbabel.OBMol:
        pyscf_mol = ob_2_pyscf(input, basis)
    elif type(input) is rdkit.Chem.rdchem.Mol:
        pyscf_mol = rdkit_2_pyscf(input, basis)
    elif type(input) is str:
        pyscf_mol = smiles_2_pyscf(input, basis)
    else:
        raise Exception('Unsupported input type.')

    return pyscf_mol

def ob_2_pyscf(mol_ob, basis):
    atom_coords = []
    for atom in ob.OBMolAtomIter(mol_ob):
        x, y, z = atom.GetX(), atom.GetY(), atom.GetZ()
        atom_type = atom.GetAtomicNum()
        atom_coords.append((atom_type, (x, y, z)))

    mol_pyscf = gto.M(atom=atom_coords, basis=basis)

    return mol_pyscf

def smiles_2_pyscf(smiles, basis):
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)

    AllChem.EmbedMolecule(rdkit_mol)
    ff = AllChem.MMFFGetMoleculeForceField(rdkit_mol, AllChem.MMFFGetMoleculeProperties(rdkit_mol))
    ff.Minimize()
    optimized_coords = rdkit_mol.GetConformer().GetPositions()

    atoms = rdkit_mol.GetAtoms()
    num_atoms = rdkit_mol.GetNumAtoms()
    pyscf_mol = gto.Mole()
    pyscf_mol.atom = []
    for i in range(num_atoms):
        atom = atoms[i]
        x, y, z = optimized_coords[i]
        atom_symbol = atom.GetSymbol()
        pyscf_mol.atom.append([atom_symbol, x, y, z])
    pyscf_mol.build(basis=basis)

    return pyscf_mol

def rdkit_2_pyscf(rdkit_mol, basis):
    coords = rdkit_mol.GetConformer().GetPositions()
    atoms = rdkit_mol.GetAtoms()
    num_atoms = rdkit_mol.GetNumAtoms()
    pyscf_mol = gto.Mole()
    pyscf_mol.atom = []
    for i in range(num_atoms):
        atom = atoms[i]
        x, y, z = coords[i]
        atom_symbol = atom.GetSymbol()
        pyscf_mol.atom.append([atom_symbol, x, y, z])
    pyscf_mol.build(basis=basis)

    return pyscf_mol