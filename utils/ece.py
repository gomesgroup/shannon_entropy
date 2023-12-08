from rdkit import Chem
from Auto3D.auto3D import options, smiles2mols
import numpy as np
import pandas as pd
from shannon import shannon_entropy, energy_and_entropy
from pyscf import gto

BOLTZ_CONST = 3.1668E-6 # value of Boltzmann constant in Hartrees (https://sciencenotes.org/boltzmann-constant-definition-and-units/)

def conf_to_pyscf(mol, basis):
    conf = mol.GetConformer()

    conf_xyz = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i).GetSymbol()
        pos = conf.GetAtomPosition(i)
        x, y, z = pos.x, pos.y, pos.z
        conf_xyz.append((atom, x, y, z))
    mol_pyscf = gto.M(atom=conf_xyz, basis=basis)

    return mol_pyscf

def generate_conformers(smiles, basis, max_confs=1000, use_gpu=False):
    args = options(k=max_confs, use_gpu=use_gpu)
    mols = smiles2mols([smiles], args)

    conf_list = []
    for mol in mols:
        conf_list.append(conf_to_pyscf(mol, basis))
    
    return conf_list

def generate_conf_df(smiles, basis, xc):
    conformers = generate_conformers(smiles=smiles, basis=basis)

    conf_df = pd.DataFrame({'SMILES': [], 'conf_num': [], 'energy': [], 'shannon_entropy': []})
    for conf_num in range(len(conformers)):
        conf_mol = conformers[conf_num]
        energy, shannon_entropy = energy_and_entropy(conf_mol, xc)

        entry = pd.DataFrame({
            'SMILES': [smiles],
            'conf_num': [conf_num],
            'energy': [energy], 
            'shannon_entropy': [shannon_entropy]
        })

        conf_df = pd.concat([conf_df, entry])
        conf_df.reset_index(drop=True, inplace=True)
    
    return conf_df

def calculate_weights(conf_df, temp):
    C = np.mean(conf_df.energy, axis=0) / (BOLTZ_CONST * temp) # offset constant C avoids overflow error
    
    # Boltzmann distribution: exp(-E / (kB * T)) / Q
    weights_unnorm = np.exp(- conf_df.energy / (BOLTZ_CONST * temp) + C)
    weights = weights_unnorm / sum(weights_unnorm)

    return weights

def calculate_ece(smiles, basis, xc, temp=298.15):
    conf_df = generate_conf_df(smiles, basis, xc)
    weights = calculate_weights(conf_df, temp)

    return sum(conf_df.shannon_entropy * weights)

def calculate_sum_ece(smiles_list, basis, xc):
    total_ece = 0
    for smiles in smiles_list:
        total_ece += calculate_ece(smiles, basis, xc)

    return total_ece

def calculate_delta_ece(reactants, products, basis, xc):
    reactants_ece = calculate_sum_ece(reactants, basis, xc)
    products_ece = calculate_sum_ece(products, basis, xc)

    return products_ece - reactants_ece
