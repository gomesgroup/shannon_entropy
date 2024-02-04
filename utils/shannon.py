from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, dft
from pyscf.tools import cubegen
import numpy as np
from convert import convert_pyscf

def calc_coper(input, basis, xc, l=200, outfile="out.cube"):
    mol = convert_pyscf(input, basis)

    mf = dft.RKS(mol)
    mf.xc = xc
    energy = mf.kernel()

    density = mf.make_rdm1()
    cube = cubegen.density(mol, outfile, density, nx=l, ny=l, nz=l)

    z = np.sum(cube)    # partition function
    cube /= z           # normalize densities
    entropy = -np.sum(cube * np.log(cube))   

    coper = np.exp(entropy - (3 * np.log(l)))

    return coper