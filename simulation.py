from autode import Reaction, Molecule
from autode.config import Config
from autode.wrappers.XTB import XTB
import os
import subprocess
import sys

def check_xtb_availability():
    """Check if XTB is available in the system"""
    try:
        result = subprocess.run(['which', 'xtb'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None

def validate_molecule(mol):
    """Validate molecule structure"""
    if not mol.atoms or len(mol.atoms) == 0:
        raise ValueError("No atoms found in molecule")
    
    print(f"\nMolecule validation for {mol.name}:")
    print(f"Number of atoms: {len(mol.atoms)}")
    print(f"Charge: {mol.charge}")
    print(f"Multiplicity: {mol.mult}")
    
    # Check for obviously wrong geometries
    for atom in mol.atoms:
        if not hasattr(atom, 'coord') or atom.coord is None:
            raise ValueError(f"Missing coordinates for atom {atom}")

def validate_optimization(mol, method):
    """Validate that a molecule has been properly optimized"""
    if not hasattr(mol, 'energy') or mol.energy is None:
        print(f"Attempting to re-optimize {mol.name}...")
        try:
            mol.optimise(method=method)
            if mol.energy is None:
                raise ValueError(f"Failed to optimize {mol.name}: Energy is None after optimization")
            print(f"Successfully optimized {mol.name}. Energy: {mol.energy}")
        except Exception as e:
            raise ValueError(f"Failed to optimize {mol.name}: {str(e)}")

def simulate_reaction(reactant_smiles, product_smiles, name="reaction"):
    """
    Simulate a chemical reaction from SMILES strings and save xyz files of the path
    """
    # Check XTB availability
    xtb_path = check_xtb_availability()
    if not xtb_path:
        raise RuntimeError("XTB executable not found in system path. Please install XTB and make sure it's in your PATH.")
    
    # Configure autodE with more detailed settings
    Config.G16.available = False
    Config.ORCA.available = False
    Config.XTB.available = True
    Config.XTB.path = xtb_path
    
    # More detailed configuration
    Config.num_conformers = 1
    Config.rmsd_threshold = 0.3
    Config.max_atom_displacement = 1.0
    Config.n_cores = 1  # Specify number of cores explicitly
    
    print(f"Using XTB from: {Config.XTB.path}")
    
    try:
        # Create molecule objects with explicit charge and multiplicity
        reactant = Molecule(smiles=reactant_smiles[0], name='reactant', charge=0, mult=1)
        product = Molecule(smiles=product_smiles[0], name='product', charge=0, mult=1)
        
        # Validate structures
        validate_molecule(reactant)
        validate_molecule(product)
        
        # Create XTB method instance with specific settings
        xtb = XTB()
        
        # Generate and optimize conformers with better error handling
        print("\nOptimizing reactant...")
        reactant.populate_conformers(n_confs=1)
        for conf in reactant.conformers:
            try:
                conf.optimise(method=xtb)
            except Exception as e:
                print(f"Error optimizing reactant conformer: {str(e)}")
                raise
        
        print("\nOptimizing product...")
        product.populate_conformers(n_confs=1)
        for conf in product.conformers:
            try:
                conf.optimise(method=xtb)
            except Exception as e:
                print(f"Error optimizing product conformer: {str(e)}")
                raise
        
        # Validate optimization and energy
        print("\nValidating optimizations...")
        validate_optimization(reactant, xtb)
        validate_optimization(product, xtb)
        
        # Store energies before conversion
        reactant_energy = reactant.energy
        product_energy = product.energy
        
        print(f"\nInitial energies before conversion:")
        print(f"Reactant energy: {reactant_energy}")
        print(f"Product energy: {product_energy}")
        
        # Convert to Reactant and Product types
        reactant = reactant.to_reactant()
        product = product.to_product()
        
        # Ensure energies are preserved after conversion
        reactant.energy = reactant_energy
        product.energy = product_energy
        
        print(f"\nEnergies after conversion:")
        print(f"Reactant energy: {reactant.energy}")
        print(f"Product energy: {product.energy}")
        
        # Create the reaction object
        reaction = Reaction(reactant, product, name=name)
        reaction.type = 'unimolecular'
        
        print("\nVerifying geometries before reaction profile...")
        if reactant.energy is None:
            raise ValueError("Reactant energy lost during conversion")
        if product.energy is None:
            raise ValueError("Product energy lost during conversion")
        
        print("\nCalculating reaction profile...")
        try:
            # Set the method before calling locate_transition_state
            reaction.method = xtb
            
            # Add more detailed settings for transition state search
            Config.ts_template_folder_path = None  # Don't use templates
            Config.min_step_size = 0.1  # Smaller step size for better convergence
            Config.max_step_size = 0.2
            
            # Add verbose output
            print("Starting transition state search...")
            reaction.locate_transition_state()
            print("Transition state search completed successfully")
            
        except Exception as e:
            print(f"Detailed error during reaction profile calculation:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            if hasattr(reaction, 'ts') and reaction.ts is not None:
                print(f"TS energy: {reaction.ts.energy}")
            raise
        
        # Save results
        os.makedirs('reaction_path', exist_ok=True)
        xyz_files = []
        
        if hasattr(reaction, 'reactant'):
            r_path = f'reaction_path/{name}_reactants.xyz'
            reaction.reactant.save_xyz_file(r_path)
            xyz_files.append(r_path)
        
        if reaction.transition_states is not None:
            for i, ts in enumerate(reaction.transition_states):
                ts_path = f'reaction_path/{name}_ts_{i}.xyz'
                ts.save_xyz_file(ts_path)
                xyz_files.append(ts_path)
        
        if hasattr(reaction, 'product'):
            p_path = f'reaction_path/{name}_products.xyz'
            reaction.product.save_xyz_file(p_path)
            xyz_files.append(p_path)
        
        return xyz_files
    
    except Exception as e:
        print(f"\nError during simulation: {str(e)}", file=sys.stderr)
        raise

def validate_energies(reaction):
    """Validate reaction energies at each step"""
    print("\nValidating reaction energies:")
    print(f"Reactant energy: {reaction.reactant.energy}")
    print(f"Product energy: {reaction.product.energy}")
    if hasattr(reaction, 'ts') and reaction.ts is not None:
        print(f"TS energy: {reaction.ts.energy}")
    
    if reaction.reactant.energy is None or reaction.product.energy is None:
        raise ValueError("Missing reactant or product energies")

def check_geometry(mol, stage):
    """Check geometry at different stages of the calculation"""
    print(f"\nGeometry check at {stage}:")
    if hasattr(mol, 'atoms') and mol.atoms:
        print(f"Number of atoms: {len(mol.atoms)}")
        print(f"Energy: {mol.energy}")
        # Check for any very short or long bonds
        for i, atom1 in enumerate(mol.atoms):
            for j, atom2 in enumerate(mol.atoms[i+1:], i+1):
                dist = ((atom1.coord - atom2.coord)**2).sum()**0.5
                if dist < 0.7 or dist > 3.0:  # Å
                    print(f"Warning: Unusual distance {dist:.2f} Å between atoms {i} and {j}")

if __name__ == "__main__":
    reactant_smiles = ["CCC(CC=C)C=C"]
    product_smiles = ["CC/C=C/CCC=C"]
    
    try:
        xyz_files = simulate_reaction(reactant_smiles, product_smiles, "simple_cyclization")
        print("Generated XYZ files:", xyz_files)
    except Exception as e:
        print(f"Simulation failed: {str(e)}")