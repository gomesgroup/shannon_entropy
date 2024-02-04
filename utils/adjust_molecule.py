from pyscf import gto
import numpy as np
import math
from openbabel import openbabel as ob

def init_mol_ob(mol_name):
    obConversion = ob.OBConversion()
    obConversion.SetInFormat("mol")
    mol_ob = ob.OBMol()
    obConversion.ReadFile(mol_ob, str.format("mol/{name}.mol", name=mol_name))

    return mol_ob

def get_bond_length(mol_name, idx_1, idx_2):
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

def adjust_bond_length(mol_name, idx_1, idx_2, length):
    eqb_length = get_bond_length(mol_name, idx_1, idx_2)
    delta_length = length - eqb_length
    
    mol_ob = init_mol_ob(mol_name)
    
    atom1 = mol_ob.GetAtom(idx_1)
    atom2 = mol_ob.GetAtom(idx_2)

    coord1 = get_coord(atom1)
    coord2 = get_coord(atom2)
    vec = coord2 - coord1
    direction = vec / np.linalg.norm(vec)

    # push atom 2 and its descendants in the same direction
    descendant_idx_forward = get_descendants(atom2, atom1)
    for idx in descendant_idx_forward:
        adjust_atom_distance(mol_ob.GetAtom(idx), direction, delta_length / 2)

    descendant_idx_backward = get_descendants(atom1, atom2)
    for idx in descendant_idx_backward:
        adjust_atom_distance(mol_ob.GetAtom(idx), direction, - delta_length / 2)

    return mol_ob

def adjust_bond_angle(mol_name, idx_1, idx_2, idx_3, angle):
    eqb_angle = get_bond_angle(mol_name, idx_1, idx_2, idx_3)
    delta_angle = angle - eqb_angle

    mol_ob = init_mol_ob(mol_name)
    
    atom1 = mol_ob.GetAtom(idx_1)
    atom2 = mol_ob.GetAtom(idx_2)
    atom3 = mol_ob.GetAtom(idx_3)

    coord1 = get_coord(atom1)
    coord2 = get_coord(atom2)
    coord3 = get_coord(atom3)

    # rotate both atoms and their respective descendants around the same axis
    axis_normalized = calculate_axis_normalized(coord1, coord2, coord3)
    rotation_matrix_forward = calculate_rotation_matrix(axis_normalized, delta_angle / 2)
    rotation_matrix_backward = calculate_rotation_matrix(axis_normalized, - delta_angle / 2)
    
    descendant_idx_forward = get_descendants(atom3, atom2)
    for idx in descendant_idx_forward:
        adjust_atom_angle(mol_ob.GetAtom(idx), rotation_matrix_forward, coord2)

    descendant_idx_backward = get_descendants(atom1, atom2)
    for idx in descendant_idx_backward:
        adjust_atom_angle(mol_ob.GetAtom(idx), rotation_matrix_backward, coord2)

    return mol_ob

def adjust_dihedral_angle(mol_name, idx_1, idx_2, idx_3, idx_4, dihedral_angle):
    mol_ob = init_mol_ob(mol_name)

    mol_ob.SetTorsion(idx_1, idx_2, idx_3, idx_4, math.radians(dihedral_angle))

    return mol_ob

def adjust_atom_distance(atom, direction, distance):
    coord = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
    new_coord = coord + direction * distance
    atom.SetVector(new_coord[0], new_coord[1], new_coord[2])

def calculate_axis_normalized(coord1, coord2, coord3):
    coord21 = coord1 - coord2
    coord23 = coord3 - coord2

    coord21_normalized = coord21 / np.linalg.norm(coord21)
    coord23_normalized = coord23 / np.linalg.norm(coord23)

    axis = np.cross(coord21_normalized, coord23_normalized)
    axis_normalized = axis / np.linalg.norm(axis)

    return axis_normalized

def calculate_rotation_matrix(axis_normalized, delta_angle):
    delta_angle_rad = np.radians(delta_angle)  # Convert angle to radians

    cos_angle = np.cos(delta_angle_rad)
    sin_angle = np.sin(delta_angle_rad)
    K = np.array([[0, -axis_normalized[2], axis_normalized[1]],
                [axis_normalized[2], 0, -axis_normalized[0]],
                [-axis_normalized[1], axis_normalized[0], 0]])
    
    rotation_matrix = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)

    return rotation_matrix

def adjust_atom_angle(atom, rotation_matrix, coord_pivot):
    coord_atom = get_coord(atom)
    coord_diff = coord_atom - coord_pivot
    coord_diff_normalized = coord_diff / np.linalg.norm(coord_diff)

    coord_diff_rotated = np.dot(rotation_matrix, coord_diff_normalized) * np.linalg.norm(coord_diff)
    coord_new = coord_pivot + coord_diff_rotated

    atom.SetVector(coord_new[0], coord_new[1], coord_new[2])

def get_descendants(curr_atom, prev_atom):
    def get_descendants_helper(curr_atom, prev_atom, descendant_idx):
        if curr_atom.GetIdx() not in descendant_idx:
            descendant_idx.add(curr_atom.GetIdx())
            for nbr_atom in ob.OBAtomAtomIter(curr_atom):
                if nbr_atom.GetIdx() != prev_atom.GetIdx():
                    descendant_idx.add(nbr_atom.GetIdx())
                    get_descendants_helper(nbr_atom, prev_atom, descendant_idx)
    
    descendant_idx = set()

    get_descendants_helper(curr_atom, prev_atom, descendant_idx)

    return descendant_idx

def get_coord(atom):
    return np.array([atom.GetX(), atom.GetY(), atom.GetZ()])