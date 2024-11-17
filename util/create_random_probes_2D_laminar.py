
"""
Created on Wed Aug  7 10:37:05 2024

@author: zippi
"""

#how to creat random points

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rand_points(minx,maxx,miny,maxy,ndata,dim):
    point = np.zeros((ndata,dim+1))
    for i in range(ndata):
        a = random.uniform(minx,maxx)
        b = random.uniform(miny,maxy)
        if a>=4.5 and a<=5.5:
            b = random.uniform(1,maxy)
        else: 
            point[i,0] = a
            point[i,1] = b
            point[i,2] = 0.5
        

    return pd.DataFrame(point).to_excel("output_2d_Lamin.xlsx",index_label=None)
    


rand_points(0, 10, 0, 5, 150, 2)
