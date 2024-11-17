# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 02:34:45 2024

@author: Amirreza
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
import time

file_name = r"E:/FOAM_PINN/cavHeat/twoD_lamin_over_box/2D_contur_Box.csv"

df = pd.read_csv(file_name)

x_ct=df[['x']].values
y_ct=df[['y']].values
u_ct=df[['u']].values

lb = [x_ct.min()  , y_ct.min()]
ub = [x_ct.max()  , y_ct.max()]

x_sec = 37
y_sec = 13
x = np.linspace(lb[0] , ub[0] , x_sec)
y = np.linspace(lb[1] , ub[1] , y_sec)

X , Y = np.meshgrid(x , y)
#u_ct = u_ct.reshape( y_sec  , x_sec)

fig, ax= plt.subplots(nrows=2 , ncols=2 , figsize=(20, 15))

fig.dpi = 300
ax[0 , 0].plot(x_ct, np.log(x_ct) * y_ct)
ax[0 , 0].set_title('Sharing Y axis')
ax[0 ,1].plot(x_ct, np.exp(x_ct)*y_ct)
ax[0 , 1].set_title('Exponential')
ax[1 , 1].set_title("Sin")
ax[1 , 1].plot(x_ct , np.sin(x_ct) * y_ct) 
plt.savefig("Comparison_plot" + time.strftime("%Y-%m-%d %H%M%S") + ".png")

