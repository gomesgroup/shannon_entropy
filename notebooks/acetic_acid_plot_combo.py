from matplotlib import pyplot as plt
import numpy as np
import math
import sys

sys.path.append("../utils")

from shannon import shannon_entropy
from adjust_molecule import convert_ob_pyscf, init_mol_ob

basis = 'sto-3g'
xc = 'B3LYP'
num_points = 100

ch3_angle = np.linspace(0.01, 120, num=num_points)
oh_angle = np.linspace(0.01, 180, num=num_points)
Z = np.zeros((len(oh_angle), len(ch3_angle)))

for i in range(len(ch3_angle)):
    for j in range(len(oh_angle)):
        mol_ob = init_mol_ob('acetic_acid')
        mol_ob.SetTorsion(2, 1, 5, 6, math.radians(ch3_angle[i]))   # twist CH3
        mol_ob.SetTorsion(2, 1, 3, 4, math.radians(oh_angle[j]))    # twist OH
        Z[j][i] = shannon_entropy(convert_ob_pyscf(mol_ob, basis=basis), xc=xc)
    
X, Y = np.meshgrid(ch3_angle, oh_angle)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Dihedral angle (CH3)')
ax.set_ylabel('Dihedral angle (OH)')
ax.set_zlabel('Shannon entropy')
plt.savefig(str.format("../plots/dihedral_angle/{mol}_{group}_dihedralangle_{min}_{max}_{num}.png", mol='acetic_acid', group='ch3oh', min=0.01, max=360, num=num_points))
plt.close()