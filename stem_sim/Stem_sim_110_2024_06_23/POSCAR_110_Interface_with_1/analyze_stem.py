from abtem import *
from ase.io import read
import matplotlib.pyplot as plt

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import savemat


filepath = '/Users/tarawork/worklocal/postdoc_lbl/Theory/theory_character/Delta_theory/stem_sim/Stem_sim_110_2024_06_23/POSCAR_110_Interface_with_1/'
exit_wave = from_zarr(filepath + 'Flexible.zarr')
exit_wave_integrate = exit_wave.integrate((100,131))
exit_wave_blurred = exit_wave_integrate.gaussian_filter(0.4)
exit_wave_integrate_numpy = exit_wave_blurred.compute()

fig, ax = plt.subplots(1, 1)
ax.imshow(exit_wave_integrate_numpy.array, cmap='magma')
fig.savefig('Stemsim_samp_0p5.png')
data = {'array': np.array([exit_wave_integrate_numpy.array])}


savemat('/Users/tarawork/worklocal/postdoc_lbl/Theory/theory_character/Delta_theory/stem_sim/Stem_sim_110_2024_06_23/POSCAR_110_Interface_with_1/domain_0p4.mat', data)

