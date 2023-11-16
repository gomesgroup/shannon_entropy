from matplotlib import pyplot as plt
import numpy as np
from utils.shannon import shannon_entropy
from utils.adjust_molecule import get_dihedral_angle, adjust_dihedral_angle, convert_ob_pyscf
from utils.adjust_molecule import init_mol_ob
import math

def plot_dihedral_angle(mol_name, group, angle_min, angle_max, num_points, idx_1, idx_2, idx_3, idx_4, basis='sto-3g', xc='B3LYP'):
    x = np.linspace(angle_min, angle_max, num=num_points)
    y = np.zeros(len(x))

    for i in range(len(x)):
        print(i)
        mol_ob = adjust_dihedral_angle(mol_name,idx_1, idx_2, idx_3, idx_4, x[i])
        y[i] = shannon_entropy(convert_ob_pyscf(mol_ob, basis=basis), xc=xc)
        
    plt.xlabel("Dihedral angle")
    plt.ylabel("Shannon entropy")
    plt.axvline(x=get_dihedral_angle(mol_name, idx_1, idx_2, idx_3, idx_4), color='r')
    plt.plot(x, y)
    plt.savefig(str.format("../plots/dihedral_angle/{mol}_{group}_dihedralangle_{min}_{max}_{num}.png", mol=mol_name, group=group, min=angle_min, max=angle_max, num=num_points))
    plt.close()

basis = 'sto-3g'
xc = 'B3LYP'
num_points = 100

ch3_angle = np.linspace(0.01, 120, num=num_points)
oh_angle = np.linspace(0.01, 180, num=num_points)
Z = np.zeros((len(ch3_angle), len(oh_angle)))

for i in range(len(ch3_angle)):
    for j in range(len(oh_angle)):
        mol_ob = init_mol_ob('acetic_acid')
        mol_ob.SetTorsion(2, 1, 5, 6, math.radians(ch3_angle[i]))   # twist CH3
        mol_ob.SetTorsion(2, 1, 3, 4, math.radians(oh_angle[j]))    # twist OH
        Z[i][j] = shannon_entropy(convert_ob_pyscf(mol_ob, basis=basis), xc=xc)
    
X, Y = np.meshgrid(ch3_angle, oh_angle)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Dihedral angle (CH3)')
ax.set_ylabel('Dihedral angle (OH)')
ax.set_zlabel('Shannon entropy');
plt.savefig(str.format("../plots/dihedral_angle/{mol}_{group}_dihedralangle_{num}.png", mol='acetic_acid', group='ch3oh', num=num_points))
plt.close()