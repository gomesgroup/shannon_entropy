import pyscf
from pyscf import gto
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import openbabel as ob

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