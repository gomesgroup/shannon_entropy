from pyscf import gto, dft
from pyscf.tools import cubegen
import numpy as np

def energy_and_entropy(mol, xc, outfile="out.cube"):
    mf = dft.RKS(mol)
    mf.xc = xc
    energy = mf.kernel()

    density = mf.make_rdm1()
    cube = cubegen.density(mol, outfile, density)

    z = np.sum(cube)    # partition function
    cube /= z           # normalize densities
    entropy = -np.sum(cube * np.log(cube))   

    return energy, entropy

def shannon_entropy(mol, xc, outfile="out.cube"):
    return energy_and_entropy(mol, xc, outfile)[1]