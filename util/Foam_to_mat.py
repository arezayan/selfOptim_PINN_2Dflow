# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 00:30:32 2024

@author: Amirreza
"""

import numpy as np
import scipy.io as io
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
# Example: Load data from a CSV file
data=pd.read_csv('data_2D_Lamin.csv', delimiter=',')


# Convert data to a dictionary format for MATLAB
mat_data = {'x': data['x'],'y': data['y'],'z': data['z'],
            'u': data['u'],'v': data['v'],'w': data['w'],'T':data['T']}
"""
# Save to .mat file
io.savemat('data_2D_Lamin.mat', mat_data)

mat_data = io.loadmat("data_2D_Lamin.mat")
#plt.plot(mat_data["u"][:,:100].flatten())
"""




x= mat_data["x"][:,:].flatten()    
y= mat_data["y"][:,:].flatten()
u= mat_data["T"][:,:].flatten()


plt.tricontourf(x, y, u)
plt.colorbar()
plt.show()