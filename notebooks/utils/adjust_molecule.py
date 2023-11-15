from pyscf import gto
import numpy as np
from matplotlib import pyplot as plt
import math
from openbabel import openbabel as ob

def init_mol_ob(mol_name):
    obConversion = ob.OBConversion()
    obConversion.SetInFormat("mol")
    mol_ob = ob.OBMol()
    obConversion.ReadFile(mol_ob, str.format("mol/{name}.mol", name=mol_name))

    return mol_ob

def get_bond_distance(mol_name, idx_1, idx_2):
    mol_ob = init_mol_ob(mol_name)

    atom1 = mol_ob.GetAtom(idx_1)
    atom2 = mol_ob.GetAtom(idx_2)

    distance = atom1.GetDistance(atom2)

    return distance

def get_bond_angle(mol_name, idx_1, idx_2, idx_3):
    mol_ob = init_mol_ob(mol_name)

    atom1 = mol_ob.GetAtom(idx_1)
    atom2 = mol_ob.GetAtom(idx_2)
    atom3 = mol_ob.GetAtom(idx_3)

    bond_angle = mol_ob.GetAngle(atom1, atom2, atom3)

    return bond_angle

def get_dihedral_angle(mol_name, idx_1, idx_2, idx_3, idx_4):
    mol_ob = init_mol_ob(mol_name)

    atom1 = mol_ob.GetAtom(idx_1)
    atom2 = mol_ob.GetAtom(idx_2)
    atom3 = mol_ob.GetAtom(idx_3)
    atom4 = mol_ob.GetAtom(idx_4)

    dihedral_angle = mol_ob.GetTorsion(atom1, atom2, atom3, atom4)

    return dihedral_angle

def adjust_bond_distance(mol_name, idx_1, idx_2, distance):
    mol_ob = init_mol_ob(mol_name)
    
    atom1 = mol_ob.GetAtom(idx_1)
    atom2 = mol_ob.GetAtom(idx_2)

    coord1 = np.array([atom1.GetX(), atom1.GetY(), atom1.GetZ()])
    coord2 = np.array([atom2.GetX(), atom2.GetY(), atom2.GetZ()])

    vec = coord2 - coord1
    current_distance = np.linalg.norm(vec)
    direction = vec / current_distance
    new_coord2 = coord1 + direction * distance

    atom2.SetVector(new_coord2[0], new_coord2[1], new_coord2[2])

    return mol_ob

def adjust_bond_angle(mol_name, idx_1, idx_2, idx_3, delta_angle):
    mol_ob = init_mol_ob(mol_name)
    
    atom1 = mol_ob.GetAtom(idx_1)
    atom2 = mol_ob.GetAtom(idx_2)
    atom3 = mol_ob.GetAtom(idx_3)

    coord1 = np.array([atom1.GetX(), atom1.GetY(), atom1.GetZ()])
    coord2 = np.array([atom2.GetX(), atom2.GetY(), atom2.GetZ()])
    coord3 = np.array([atom3.GetX(), atom3.GetY(), atom3.GetZ()])

    coord21 = coord1 - coord2
    coord23 = coord3 - coord2

    coord21_normalized = coord21 / np.linalg.norm(coord21)
    coord23_normalized = coord23 / np.linalg.norm(coord23)

    axis = np.cross(coord21_normalized, coord23_normalized)
    axis_normalized = axis / np.linalg.norm(axis)
    delta_angle_rad = np.radians(delta_angle)  # Convert angle to radians

    cos_angle = np.cos(delta_angle_rad)
    sin_angle = np.sin(delta_angle_rad)
    K = np.array([[0, -axis_normalized[2], axis_normalized[1]],
                [axis_normalized[2], 0, -axis_normalized[0]],
                [-axis_normalized[1], axis_normalized[0], 0]])
    rotation_matrix = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)

    coord23_rotated = np.dot(rotation_matrix, coord23_normalized) * np.linalg.norm(coord23)
    coord3_new = coord2 + coord23_rotated

    atom3.SetVector(coord3_new[0], coord3_new[1], coord3_new[2])

    return mol_ob

def adjust_dihedral_angle(mol_name, idx_1, idx_2, idx_3, idx_4, dihedral_angle):
    mol_ob = init_mol_ob(mol_name)

    mol_ob.SetTorsion(idx_1, idx_2, idx_3, idx_4, math.radians(dihedral_angle))

    return mol_ob

def adjust_bond_distance_extended(mol_name, idx_1, idx_2, distance):
    mol_ob = init_mol_ob(mol_name)
    
    atom1 = mol_ob.GetAtom(idx_1)
    atom2 = mol_ob.GetAtom(idx_2)

    coord1 = np.array([atom1.GetX(), atom1.GetY(), atom1.GetZ()])
    coord2 = np.array([atom2.GetX(), atom2.GetY(), atom2.GetZ()])
    vec = coord2 - coord1
    direction = vec / np.linalg.norm(vec)

    adjust_atom_distance(atom2, direction, distance)

    # push the neighbors of atom 2 in the same direction
    for bond in ob.OBAtomBondIterm(atom2):
        nbr_atom = bond.GetNbrAtom(atom2)
        if nbr_atom.GetIdx() != idx_1:
            adjust_atom_distance(nbr_atom, direction, distance)

    return mol_ob

def adjust_atom_distance(atom, direction, distance):
    coord = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
    new_coord = coord + direction * distance
    atom.SetVector(new_coord[0], new_coord[1], new_coord[2])

def convert_ob_pyscf(mol_ob, basis):
    atom_coords = []
    for atom in ob.OBMolAtomIter(mol_ob):
        x, y, z = atom.GetX(), atom.GetY(), atom.GetZ()
        atom_type = atom.GetAtomicNum()
        atom_coords.append((atom_type, (x, y, z)))

    mol_pyscf = gto.M(atom=atom_coords, basis=basis)

    return mol_pyscf