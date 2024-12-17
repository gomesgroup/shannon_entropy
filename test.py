import autode as ade

h2 = ade.Molecule(smiles='[H][H]')
h2.optimise(method=ade.methods.ORCA())