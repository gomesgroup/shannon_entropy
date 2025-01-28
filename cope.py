import subprocess
import autode as ade
import os
import argparse
import logging
from information_profile import compute_information_profile

def parse_args():
    parser = argparse.ArgumentParser(description='Run Cope reaction calculation with specified X group')
    parser.add_argument('X', type=str, help='Substituent group for the reaction (e.g., Cl for chlorine)')
    parser.add_argument('--basis', type=str, help='Basis set')
    parser.add_argument('--xc', type=str, help='Exchange-correlation functional')
    parser.add_argument('--nprocs', type=int, default=32, help='Number of processors')
    parser.add_argument('--mem', type=int, default=4000, help='Memory usage')
    parser.add_argument('--l', type=int, help='Grid resolution')
    return parser.parse_args()

def setup_cope(dir, X):
    # Set up the reaction
    reactant_smiles = f'{X}/C({X})=C/CCC=C'
    product_smiles = f'C=CC({X})(CC=C){X}'

    r = ade.Reactant(name='reactant', smiles=reactant_smiles)
    p = ade.Product(name='product', smiles=product_smiles)
    reaction = ade.Reaction(r, p, name=dir)
    
    return reaction

def write_irc_input(irc_input_file, reaction, nprocs):
    # Obtain transition state coordinates
    ts = reaction.ts
    coords = ts.coordinates
    atoms = ts.atoms

    # Create the input file header
    header = f"""!r2SCAN-3c IRC 

%pal
  nprocs {nprocs}
end

%IRC
 MAXITER 30
END

* xyz 0 1
"""
    
    # Create the coordinate block
    coord_block = ""
    for atom, pos in zip(atoms, coords):
        coord_block += f"{atom.label:2s} {pos[0]:12.5f} {pos[1]:12.5f} {pos[2]:12.5f}\n"
    with open(irc_input_file, 'w') as f:
        f.write(header)
        f.write(coord_block)
        f.write('*\n') 

if __name__ == "__main__":
    # ------------------------------------------------
    # Step 0: Set up logging and autodE
    # ------------------------------------------------

    # Parse arguments
    args = parse_args()
    X = args.X
    basis = args.basis
    xc = args.xc
    l = int(args.l)
    nprocs = args.nprocs
    mem = args.mem

    # print PID and settings
    print(os.getpid())
    print(f"X: {X}")
    print(f"Basis set: {basis}")
    print(f"Exchange-correlation functional: {xc}")
    print(f"Grid resolution: {l}")
    print(f"Number of processors: {nprocs}")
    print(f"Memory usage: {mem}")

    # Make directory
    dir = os.path.join("reactions", "cope", X)
    os.makedirs(dir, exist_ok=True)

    # Set up logging
    ade.log.logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join(dir, f'log.log'))
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.INFO)
    
    # Configure ORCA with appropriate basis set for heavy atoms
    kwds = ade.Config.ORCA.keywords
    kwds.sp = ['SP', 'PBE0', 'def2-TZVP', 'RIJCOSX', 'def2/J']
    kwds.opt = ['Opt', 'PBE0', 'def2-TZVP', 'RIJCOSX', 'def2/J']
    kwds.opt_ts = ['OptTS', 'PBE0', 'def2-TZVP', 'RIJCOSX', 'def2/J',
                   '\n%geom\n'
                   'Calc_Hess true\n'
                   'Recalc_Hess 30\n'
                   'Trust -0.1\n'
                   'MaxIter 150\n'
                   'end']
    kwds.hess = ['Freq', 'PBE0', 'def2-TZVP', 'RIJCOSX', 'def2/J']
    kwds.grad = ['EnGrad', 'PBE0', 'def2-TZVP', 'RIJCOSX', 'def2/J']
    
    # Set up methods
    ade.Config.lcode = 'xtb'  # Use XTB directly as low-level method
    ade.Config.hcode = 'orca'  # Use ORCA as high-level method
    
    # Set resource usage (50% of system resources)
    ade.Config.n_cores = nprocs
    ade.Config.max_core = mem
    
    # Set paths
    ade.Config.ORCA.path = '/opt/orca-6.0.0/orca'
    ade.Config.XTB.path = os.path.expanduser('~/.conda/envs/comp_chem/bin/xtb')
    
    # ------------------------------------------------
    # Step 1: Run Cope reaction
    # ------------------------------------------------

    reaction = setup_cope(dir, X)
    reaction.calculate_reaction_profile()

    # ------------------------------------------------
    # Step 2: Run IRC calculations
    # ------------------------------------------------  

    # Write the complete input file
    irc_dir = os.path.join(dir, 'irc')
    os.makedirs(irc_dir, exist_ok=True)
    irc_input_file = os.path.join(irc_dir, 'irc.inp')
    write_irc_input(irc_input_file, reaction, nprocs)
    
    # Run ORCA IRC calculation
    irc_cmd = f"$(which orca) {irc_input_file}"
    subprocess.run(irc_cmd, shell=True)

    # ------------------------------------------------
    # Step 3: Obtain information profile
    # ------------------------------------------------  

    compute_information_profile(dir, basis, xc, l)