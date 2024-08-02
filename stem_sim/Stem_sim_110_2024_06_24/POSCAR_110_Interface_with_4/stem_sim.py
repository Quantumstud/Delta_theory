from abtem import __version__
print('Abtem version:',__version__)
import matplotlib.pyplot as plt
from ase.io import read
import numpy as np
from abtem import *
from ase import Atoms
from ase.geometry import wrap_positions

## Read structure

atoms = read('POSCAR_110_Interface_with_4',format='vasp')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

show_atoms(atoms, ax=ax1, title='Top view')
show_atoms(atoms, ax=ax2, plane='xz', title='Side view')
fig.savefig('atoms.jpg')
print("Lattice parameters of the original cell:",atoms.cell)

# Define the rotation matrix for 90 degrees along the x-axis
rotation_matrix = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
])

# Apply the rotation to the cell
original_cell = atoms.get_cell().array
new_cell = np.dot(rotation_matrix, original_cell.T).T

# Create the correctly ordered new cell
correctly_ordered_new_cell = [new_cell[0], -new_cell[2], new_cell[1]]

# Apply the rotation to the positions
new_positions = np.dot(atoms.get_positions(), rotation_matrix)

# Ensure the atomic positions are wrapped within the new cell
new_positions = wrap_positions(new_positions, correctly_ordered_new_cell)

# Create a new Atoms object with the rotated positions and correctly ordered cell
atoms_rotated = Atoms(symbols=atoms.get_chemical_symbols(), positions=new_positions, cell=correctly_ordered_new_cell, pbc=atoms.get_pbc())

# Plot the rotated structure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
show_atoms(atoms_rotated, ax=ax1, title='Top view (rotated)')
show_atoms(atoms_rotated, ax=ax2, plane='xz', title='Side view (rotated)')
fig.savefig('atoms_rotated.jpg')

print("Lattice parameters of the rotated cell:", atoms_rotated.cell)

# # Create potential
potential = Potential(atoms_rotated, 
                      sampling=0.05,
                      projection='finite', 
                      slice_thickness=1, 
                      parametrization='lobato',
		              device = 'gpu')

potential = potential.build()
print(f"Real space sampling: {potential.sampling}")
print("Potential GPTS",potential.gpts)


probe = Probe(energy=300e3, semiangle_cutoff=28,device='gpu')
probe.grid.match(potential)

detector = FlexibleAnnularDetector()

gridscan = GridScan(start=[0.05*atoms_rotated.get_cell_lengths_and_angles()[0], 0.05*atoms_rotated.get_cell_lengths_and_angles()[1]], end=[0.95*atoms_rotated.get_cell_lengths_and_angles()[0], 0.95*atoms_rotated.get_cell_lengths_and_angles()[1]], sampling = 0.10)
fig, ax1 = plt.subplots(1,1, figsize=(4,4))
potential.project().show(ax1)
gridscan.add_to_plot(ax1)
fig.savefig('scan.png')

flexible_measurement = probe.scan(potential,scan=gridscan,detectors=detector)
flexible_measurement.compute()
flexible_measurement.to_zarr("Flexible.zarr")
