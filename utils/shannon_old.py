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
import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, lax
# Enable double precision
jax.config.update('jax_enable_x64', True)
import struct
import os
import numpy as np
import cupy as cp
from pyscf.tools.cubegen import Cube
from functools import partial

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

def run_dft(input, basis, xc):
    mol = convert_pyscf(input, basis)
    mf = pyscf.dft.RKS(mol)
    mf.xc = xc
    energy = mf.kernel()

    return mf, energy

def compute_density_cpu(mf, mol, l):
    density = mf.make_rdm1()
    return density

def gather_cube_cpu(density, mol, l):
    cube = cpu_cubegen.density(mol, "out.cube", density, nx=l, ny=l, nz=l)
    return cube

def compute_density_gpu(mf, mol, l):
    density = mf.make_rdm1()
    return density

@partial(jit, static_argnums=(2,))
def process_batch(coords_batch, dm, mol):
    """JIT-compiled function to process a batch of coordinates"""
    ao = jnp.asarray(mol.eval_gto('GTOval', coords_batch))
    return eval_rho_jax(ao, dm)

@jit
def eval_rho_jax(ao, dm):
    """Optimized JAX GPU-accelerated version of eval_rho"""
    return jnp.einsum('pi,ij,pj->p', ao, dm, ao, optimize='optimal')

@partial(jit, static_argnums=(1,))
def compute_density_batch(batch_data, mol):
    """Compute density for multiple batches in parallel"""
    coords, dm = batch_data
    return process_batch(coords, dm, mol)

class CubeGPU(Cube):
    """Optimized JAX GPU-accelerated version of the Cube class"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._coords_gpu = None
        self._batch_size = 32000  # Optimized batch size
        
    def _cartesian_prod(self, xs, ys, zs):
        """JAX-compatible cartesian product"""
        xs = jnp.asarray(xs)
        ys = jnp.asarray(ys)
        zs = jnp.asarray(zs)
        
        xs_grid = jnp.repeat(xs[:, None, None], ys.shape[0], axis=1)
        xs_grid = jnp.repeat(xs_grid, zs.shape[0], axis=2)
        
        ys_grid = jnp.repeat(ys[None, :, None], xs.shape[0], axis=0)
        ys_grid = jnp.repeat(ys_grid, zs.shape[0], axis=2)
        
        zs_grid = jnp.repeat(zs[None, None, :], xs.shape[0], axis=0)
        zs_grid = jnp.repeat(zs_grid, ys.shape[0], axis=1)
        
        return jnp.stack([
            xs_grid.reshape(-1),
            ys_grid.reshape(-1),
            zs_grid.reshape(-1)
        ], axis=1)
        
    @partial(jit, static_argnums=(0,))
    def _compute_coords(self, xs, ys, zs, box, origin):
        """JIT-compiled coordinate computation"""
        frac_coords = self._cartesian_prod(xs, ys, zs)
        return frac_coords @ box + origin
        
    def get_coords(self):
        """Get coordinates using optimized JAX GPU acceleration"""
        if self._coords_gpu is None:
            # Convert all inputs to JAX arrays
            xs = jnp.asarray(self.xs)
            ys = jnp.asarray(self.ys)
            zs = jnp.asarray(self.zs)
            box = jnp.asarray(self.box)
            origin = jnp.asarray(self.boxorig)
            
            # Compute coordinates using JIT-compiled function
            self._coords_gpu = self._compute_coords(xs, ys, zs, box, origin)
        return self._coords_gpu

    def write(self, field, fname, comment=None):
        """Optimized cube file writing"""
        if isinstance(field, jnp.ndarray):
            field = np.array(field)
        super().write(field, fname, comment)

@jit
def process_single_point(coord, dm):
    """Process a single coordinate point"""
    ao = jnp.asarray(mol.eval_gto('GTOval', coord[None]))
    return eval_rho_jax(ao, dm)

def gather_cube_gpu(density, mol, l):
    """Highly optimized JAX GPU-accelerated cube generation"""
    from pyscf.pbc.gto import Cell
    
    # Initialize cube with GPU acceleration
    cc = CubeGPU(mol, l, l, l, RESOLUTION, BOX_MARGIN)

    # Get coordinates and convert density matrix to JAX array
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    
    if hasattr(density, 'asnumpy'):
        dm_gpu = jnp.asarray(density.asnumpy())
    elif hasattr(density, 'get'):
        dm_gpu = jnp.asarray(density.get())
    else:
        dm_gpu = jnp.asarray(density)

    # Process in chunks to avoid memory issues
    chunk_size = 8000
    n_chunks = (ngrids + chunk_size - 1) // chunk_size
    rho = np.zeros(ngrids)

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, ngrids)
        
        # Process chunk
        coords_chunk = np.array(coords[start_idx:end_idx])
        ao = jnp.asarray(mol.eval_gto('GTOval', coords_chunk))
        rho_chunk = eval_rho_jax(ao, dm_gpu)
        
        # Store results
        rho[start_idx:end_idx] = np.array(rho_chunk)

    # Convert final result to JAX array and reshape
    rho = jnp.asarray(rho).reshape((cc.nx, cc.ny, cc.nz))
    
    # Write output
    cc.write(rho, "out.cube", comment='Electron density in real space (e/Bohr^3)')
    return np.array(rho)

@jit
def compute_coper(cube, l):
    """Optimized COPER computation using JAX"""
    cube = jnp.asarray(cube)
    z = jnp.sum(cube)
    cube = cube / z
    # Use where to handle log(0) cases
    log_cube = jnp.where(cube > 0, jnp.log(cube), 0.0)
    entropy = -jnp.sum(jnp.where(cube > 0, cube * log_cube, 0.0))
    coper = jnp.exp(entropy - (3 * jnp.log(jnp.array(l, dtype=jnp.float64))))
    return jnp.array(coper, dtype=jnp.float64)

def calc_coper_cpu(input, basis, xc, l):
    mol = convert_pyscf(input, basis)

    mf = pyscf.dft.RKS(mol)
    mf.xc = xc
    energy = mf.kernel()

    density = mf.make_rdm1()
    cube = cpu_cubegen.density(mol, "out.cube", density, nx=l, ny=l, nz=l)

    z = np.sum(cube)    # partition function
    cube = cube / z      # normalize densities
    entropy = -np.sum(cube * np.log(cube))   

    coper = np.exp(entropy - (3 * np.log(l)))

    return coper, energy

def calc_coper(input, basis, xc, l):
    mol = convert_pyscf(input, basis)

    mf = rks.RKS(mol)
    mf.xc = xc
    energy = mf.kernel()

    # Get density matrix
    density = mf.make_rdm1()
    
    # Generate cube using GPU acceleration
    cube = gather_cube_gpu(density, mol, l)
    
    # Calculate COPER using JAX
    coper = float(compute_coper(cube, l))

    return coper, energy

def calc_entropy(input, basis, xc, l):
    mol = convert_pyscf(input, basis)

    mf = rks.RKS(mol)
    mf.xc = xc
    energy = mf.kernel()

    density = mf.make_rdm1()
    cube = cpu_cubegen.density(mol, "out.cube", density, nx=l, ny=l, nz=l)

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
    cc.write(rho, outfile, comment='Electron density in real space (e/Bohr^3)')
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
    
    # # Benchmark CPU version
    # print("\nCPU Version Breakdown:")
    # start_cpu = time.time()
    
    # start_dft = time.time()
    # mf_cpu, energy_cpu = run_dft_cpu(mol, basis, xc)
    # dft_time = time.time() - start_dft
    # print(f"DFT calculation: {dft_time:.3f} seconds")
    
    # start_density = time.time()
    # density_cpu = compute_density_cpu(mf_cpu, mol, l)
    # density_time = time.time() - start_density
    # print(f"Density calculation: {density_time:.3f} seconds")
    
    # start_cube = time.time()
    # cube_cpu = gather_cube_cpu(density_cpu, mol, l) 
    # cube_time = time.time() - start_cube
    # print(f"Cube generation: {cube_time:.3f} seconds")
    
    # start_coper = time.time()
    # coper_cpu = compute_coper(cube_cpu, l)
    # coper_time = time.time() - start_coper
    # print(f"COPER calculation: {coper_time:.3f} seconds")
    
    # cpu_time = time.time() - start_cpu
    # print(f"Total CPU time: {cpu_time:.3f} seconds\n")
    
    # Benchmark GPU version
    print("GPU Version Breakdown:") 
    start_gpu = time.time()
    
    start_dft = time.time()
    mf_gpu, energy_gpu = run_dft(mol, basis, xc)
    dft_time = time.time() - start_dft
    print(f"DFT calculation: {dft_time:.3f} seconds")
    
    start_density = time.time()
    density_gpu = compute_density_gpu(mf_gpu, mol, l)
    density_time = time.time() - start_density
    print(f"Density calculation: {density_time:.3f} seconds")
    
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
    if abs(coper_cpu - coper_gpu) < 1e-6:
        print("COPER values match within tolerance")
    else:
        print("Warning: COPER values differ!")
        
    if abs(energy_cpu - energy_gpu) < 1e-6:
        print("Energy values match within tolerance")
    else:
        print("Warning: Energy values differ!")