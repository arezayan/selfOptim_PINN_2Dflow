# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:37:31 2024

@author: zippi
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy .io

from matplotlib.colors import LogNorm

# Fixing random state for reproducibility
np.random.seed(19680801)

# make these smaller to increase the resolution
dx, dy = 0.15, 0.05

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[-3:3+dy:dy, -8:8+dx:dx]
z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)
# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
#z = z[:-1, :-1]
z_min, z_max = -abs(z).max(), abs(z).max()

fig, axs = plt.subplots(2, 1)

ax = axs[ 0]
c = ax.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('Exact')
fig.colorbar(c, ax=ax)

ax = axs[ 1]
c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('PINN')
fig.colorbar(c, ax=ax)

fig.tight_layout()
plt.show()



