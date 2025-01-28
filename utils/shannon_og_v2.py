import time
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pyscf
import openbabel as ob
from pyscf import gto, dft
from pyscf.tools import cubegen as cpu_cubegen
from pyscf.dft import numint
from pyscf import lib
from pyscf import df
import jax.numpy as jnp
from jax import jit
import struct
import os
import numpy as np
import cupy as cp
from pyscf.tools.cubegen import Cube

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 3 which has more free memory
from gpu4pyscf.dft import rks

RESOLUTION = None
BOX_MARGIN = 3.0
ORIGIN = None
EXTENT = None

def run_dft_cpu(input, basis, xc):
    mol = convert_pyscf(input, basis)
    mf = pyscf.dft.RKS(mol)
    mf.xc = xc
    energy = mf.kernel()

    return mf, energy

def run_dft_gpu(input, basis, xc):
    mol = convert_pyscf(input, basis)
    mf = rks.RKS(mol)
    mf.xc = xc
    energy = mf.kernel()

    return mf, energy

def gather_cube_cpu(density, mol, l):
    cube = cpu_cubegen.density(mol, "out.cube", density, nx=l, ny=l, nz=l)
    return cube

def gather_cube_gpu(density, mol, l):
    """JAX GPU-accelerated version of cube generation"""
    from pyscf.pbc.gto import Cell
    
    # Initialize cube with GPU acceleration
    cc = CubeGPU(mol, l, l, l, RESOLUTION, BOX_MARGIN)

    GTOval = 'GTOval'
    if isinstance(mol, Cell):
        GTOval = 'PBC' + GTOval

    # Get coordinates from pre-allocated GPU memory
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    
    # Optimize batch size for GPU memory and occupancy
    blksize = min(8000, ngrids)  # Reduced batch size for better precision
    
    # Convert density matrix to JAX array with explicit float64 precision
    if hasattr(density, 'asnumpy'):
        dm_gpu = jnp.asarray(density.asnumpy(), dtype=jnp.float64)
    elif hasattr(density, 'get'):
        dm_gpu = jnp.asarray(density.get(), dtype=jnp.float64)
    else:
        dm_gpu = jnp.asarray(density, dtype=jnp.float64)
    
    # Pre-allocate output array with float64 precision
    rho = jnp.zeros(ngrids, dtype=jnp.float64)
    
    # Process in batches
    for ip0 in range(0, ngrids, blksize):
        ip1 = min(ip0 + blksize, ngrids)
        
        # Convert coordinates back to CPU for eval_gto
        coords_batch = np.array(coords[ip0:ip1])
        
        # Evaluate AO and convert to JAX array with float64 precision
        ao = jnp.asarray(mol.eval_gto(GTOval, coords_batch), dtype=jnp.float64)
        
        # Compute density for this batch using JIT-compiled function
        rho = rho.at[ip0:ip1].set(eval_rho_jax(ao, dm_gpu))
    
    # Reshape result and ensure proper normalization
    rho = rho.reshape((cc.nx, cc.ny, cc.nz))
    
    # Convert back to numpy with float64 precision to match CPU behavior
    return np.array(rho, dtype=np.float64)

def compute_coper(cube, l):
    """Compute COPER using JAX for GPU acceleration with explicit float64 precision"""
    # Convert input to float64 precision
    cube = jnp.asarray(cube, dtype=jnp.float64)
    z = jnp.sum(cube)    # partition function
    cube = cube / z      # normalize densities
    # Use float64 precision for log computation
    entropy = -jnp.sum(cube * jnp.log(cube + 1e-15))   # add small epsilon to avoid log(0)
    coper = jnp.exp(entropy - (3 * jnp.log(jnp.float64(l))))
    return float(coper)

def calc_coper_cpu(input, basis, xc, l):
    mol = convert_pyscf(input, basis)

    mf, energy = run_dft_cpu(mol, basis, xc)
    density = mf.make_rdm1()
    cube = cpu_cubegen.density(mol, "out.cube", density, nx=l, ny=l, nz=l)

    coper = compute_coper(cube, l)

    return coper, energy

def calc_coper_gpu(input, basis, xc, l):
    mol = convert_pyscf(input, basis)

    mf, energy = run_dft_gpu(mol, basis, xc)
    density = mf.make_rdm1()
    cube = gather_cube_gpu(density, mol, l)
    coper = compute_coper(cube, l)

    return coper, energy

def calc_entropy_cpu(input, basis, xc, l):
    mol = convert_pyscf(input, basis)

    mf, energy = run_dft_cpu(mol, basis, xc)
    density = mf.make_rdm1()
    cube = gather_cube_cpu(density, mol, l)

    z = jnp.sum(cube)    # partition function
    cube = cube / z      # normalize densities
    entropy = -jnp.sum(cube * jnp.log(cube))   

    return entropy

def calc_entropy_gpu(input, basis, xc, l):
    mol = convert_pyscf(input, basis)

    mf, energy = run_dft_gpu(mol, basis, xc)
    density = mf.make_rdm1()
    cube = gather_cube_gpu(density, mol, l)

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

class CubeGPU(Cube):
    """JAX GPU-accelerated version of the Cube class for density calculations"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-allocate GPU memory for frequently used arrays
        self._coords_gpu = None
        
    def get_coords(self):
        """Get coordinates using JAX GPU acceleration with memory reuse"""
        if self._coords_gpu is None:
            frac_coords = jnp.asarray(lib.cartesian_prod([self.xs, self.ys, self.zs]))
            box_gpu = jnp.asarray(self.box)
            origin_gpu = jnp.asarray(self.boxorig)
            self._coords_gpu = frac_coords @ box_gpu + origin_gpu
        return self._coords_gpu

    def write(self, field, fname, comment=None):
        """Write cube file with optimized GPU to CPU transfer"""
        if isinstance(field, jnp.ndarray):
            field = np.array(field)
        super().write(field, fname, comment)

@jit
def eval_rho_jax(ao, dm):
    """JAX GPU-accelerated version of eval_rho with explicit float64 precision"""
    return jnp.einsum('pi,ij,pj->p', ao, dm, ao, precision="highest")

def density_gpu(mol, outfile, dm, nx=80, ny=80, nz=80, resolution=RESOLUTION, margin=BOX_MARGIN):
    """GPU-accelerated version of density calculation"""
    from pyscf.pbc.gto import Cell
    cc = CubeGPU(mol, nx, ny, nz, resolution, margin)

    GTOval = 'GTOval'
    if isinstance(mol, Cell):
        GTOval = 'PBC' + GTOval

    # Compute density on the .cube grid using GPU
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    rho = jnp.zeros(ngrids)
    
    # Convert density matrix to GPU
    dm_gpu = jnp.asarray(dm)
    
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = jnp.asarray(mol.eval_gto(GTOval, jnp.asnumpy(coords[ip0:ip1])))
        rho = rho.at[ip0:ip1].set(eval_rho_jax(ao, dm_gpu))
    rho = rho.reshape(cc.nx, cc.ny, cc.nz)

    # Write out density to the .cube file
    # cc.write(rho, outfile, comment='Electron density in real space (e/Bohr^3)')
    return jnp.asnumpy(rho)

def orbital_gpu(mol, outfile, coeff, nx=80, ny=80, nz=80, resolution=RESOLUTION, margin=BOX_MARGIN):
    """GPU-accelerated version of orbital calculation"""
    from pyscf.pbc.gto import Cell
    cc = CubeGPU(mol, nx, ny, nz, resolution, margin)

    GTOval = 'GTOval'
    if isinstance(mol, Cell):
        GTOval = 'PBC' + GTOval

    # Compute orbital on the .cube grid using GPU
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    orb_on_grid = jnp.zeros(ngrids)
    
    # Convert coefficient to GPU
    coeff_gpu = jnp.asarray(coeff)
    
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = jnp.asarray(mol.eval_gto(GTOval, jnp.asnumpy(coords[ip0:ip1])))
        orb_on_grid = orb_on_grid.at[ip0:ip1].set(jnp.dot(ao, coeff_gpu))
    orb_on_grid = orb_on_grid.reshape(cc.nx, cc.ny, cc.nz)

    # Write out orbital to the .cube file
    cc.write(orb_on_grid, outfile, comment='Orbital value in real space (1/Bohr^3)')
    return jnp.asnumpy(orb_on_grid)

def mep_gpu(mol, outfile, dm, nx=80, ny=80, nz=80, resolution=RESOLUTION, margin=BOX_MARGIN):
    """GPU-accelerated version of molecular electrostatic potential calculation"""
    cc = CubeGPU(mol, nx, ny, nz, resolution, margin)
    coords = cc.get_coords()
    coords_cpu = jnp.asnumpy(coords)  # Need CPU version for some operations

    # Nuclear potential at given points (on GPU)
    Vnuc = jnp.zeros(len(coords))
    for i in range(mol.natm):
        r = jnp.asarray(mol.atom_coord(i))
        Z = mol.atom_charge(i)
        rp = r - coords
        Vnuc += Z / jnp.sqrt(jnp.sum(rp * rp, axis=1))

    # Potential of electron density (on GPU)
    Vele = jnp.zeros_like(Vnuc)
    for p0, p1 in lib.prange(0, Vele.size, 600):
        fakemol = gto.fakemol_for_charges(coords_cpu[p0:p1])
        ints = jnp.asarray(df.incore.aux_e2(mol, fakemol))
        dm_gpu = jnp.asarray(dm)
        Vele = Vele.at[p0:p1].set(jnp.einsum('ijp,ij->p', ints, dm_gpu))

    MEP = Vnuc - Vele  # MEP at each point
    MEP = MEP.reshape(cc.nx, cc.ny, cc.nz)

    # Write the potential
    cc.write(MEP, outfile, 'Molecular electrostatic potential in real space')
    return jnp.asnumpy(MEP)

# Add the optimized GPU version to numint module
numint.eval_rho_gpu = eval_rho_jax

if __name__ == "__main__":
    basis = "def2-tzvp"
    xc = 'pbe0'
    l = 200

    # Test molecule
    test_smiles = "CC(=O)O"  # Acetic acid
    mol = smiles_2_pyscf(test_smiles, basis)
    
    # Benchmark CPU version
    print("\nCPU Version Breakdown:")
    start_cpu = time.time()
    
    start_dft = time.time()
    mf_cpu, energy_cpu = run_dft_cpu(mol, basis, xc)
    dft_time = time.time() - start_dft
    print(f"DFT calculation: {dft_time:.3f} seconds")
    
    density_cpu = mf_cpu.make_rdm1()
    
    start_cube = time.time()
    cube_cpu = gather_cube_cpu(density_cpu, mol, l) 
    cube_time = time.time() - start_cube
    print(f"Cube generation: {cube_time:.3f} seconds")
    
    start_coper = time.time()
    coper_cpu = compute_coper(cube_cpu, l)
    coper_time = time.time() - start_coper
    print(f"COPER calculation: {coper_time:.3f} seconds")
    
    cpu_time = time.time() - start_cpu
    print(f"Total CPU time: {cpu_time:.3f} seconds\n")
    
    # Test molecule
    test_smiles = "CC(=O)O"  # Acetic acid
    mol = smiles_2_pyscf(test_smiles, basis)

    # Benchmark GPU version
    print("GPU Version Breakdown:") 
    start_gpu = time.time()
    
    start_dft = time.time()
    mf_gpu, energy_gpu = run_dft_gpu(mol, basis, xc)
    dft_time = time.time() - start_dft
    print(f"DFT calculation: {dft_time:.3f} seconds")
    
    density_gpu = mf_gpu.make_rdm1()
    
    start_cube = time.time()
    cube_gpu = gather_cube_gpu(density_gpu, mol, l)
    cube_time = time.time() - start_cube
    print(f"Cube generation: {cube_time:.3f} seconds")
    
    start_coper = time.time()
    coper_gpu = compute_coper(cube_gpu, l)
    coper_time = time.time() - start_coper
    print(f"COPER calculation: {coper_time:.3f} seconds")
    
    gpu_time = time.time() - start_gpu
    print(f"Total GPU time: {gpu_time:.3f} seconds\n")
    
    # Summary
    print("Summary:")
    print(f"Total CPU time: {cpu_time:.3f} seconds")
    print(f"Total GPU time: {gpu_time:.3f} seconds")
    print(f"Overall speedup: {cpu_time/gpu_time:.2f}x\n")
    
    # Verify results match
    print("Validation:")
    print("COPER CPU: ", coper_cpu)
    print("COPER GPU: ", coper_gpu)
    if abs(coper_cpu - coper_gpu) < 1e-6:
        print("COPER values match within tolerance")
    else:
        print("Warning: COPER values differ!")
        
    if abs(energy_cpu - energy_gpu) < 1e-6:
        print("Energy values match within tolerance")
    else:
        print("Warning: Energy values differ!")