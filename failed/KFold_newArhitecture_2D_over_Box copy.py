# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:34:29 2024

@author: Amirreza
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import optuna
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  
import time  


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Swish(nn.Module):
        def __init__(self, inplace=True):
            super(Swish, self).__init__()
            self.inplace = inplace

        def forward(self, x):
            if self.inplace:
                x.mul_(torch.sigmoid(x))
                return x
            else:
                return x * torch.sigmoid(x)
    

class PINN_u(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(PINN_u, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),


                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),


                Swish(),

                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        #def forward(self,x):
        def forward(self,x):	
            output = self.main(x)
            return output

class PINN_v(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(PINN_v, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),


                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),

                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        #def forward(self,x):
        def forward(self,x):	
            output = self.main(x)
            return output

class PINN_w(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(PINN_w, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),


                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),

                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        #def forward(self,x):
        def forward(self,x):	
            output = self.main(x)
            return output

class PINN_p(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(PINN_p, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),

                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        def forward(self,x):
            output = self.main(x)
            return  output	

class PINN_T(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(PINN_T, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),

                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        def forward(self,x):
            output = self.main(x)
            return  output	

class PINN_phi(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(PINN_phi, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),

                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        def forward(self,x):
            output = self.main(x)
            return  output	
    

################################################################

def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)
def init_xavier(m):
    if isinstance(m, nn.Linear):  # Check if the layer is nn.Linear
        nn.init.xavier_normal_(m.weight)  # Xavier normal distribution
        if m.bias is not None:           # Initialize biases to zero if present
            nn.init.zeros_(m.bias)  


#PINN_u is model(networks foe each components)
input_n = 2
h_n = 10
PINN_u = PINN_u().to(device)
PINN_v = PINN_v().to(device)
PINN_w = PINN_w().to(device)
PINN_p = PINN_p().to(device)
PINN_T = PINN_T().to(device)
PINN_phi = PINN_phi().to(device)

PINN_u.apply(init_xavier)
PINN_v.apply(init_xavier)
PINN_p.apply(init_xavier)
PINN_T.apply(init_xavier)
PINN_phi.apply(init_xavier)


###################################################################
#functions

def normal_inputs(df): #df is a dataframe
    normal_df = (2 * (df - df.min()) / (df.max() - df.min() )) - 1
    return normal_df

def residula_loss(PINN_phi  , PINN_p , PINN_T , x , y ):
    
    x = (2 * (x - x.min()) / (x.max() - x.min())) - 1
    y = (2 * (y - y.min()) / (y.max() - y.min())) - 1
        
    #u = PINN_u(torch.cat((x,y) , dim = 1))
    #v = PINN_v(torch.cat((x,y) , dim = 1))
    phi = PINN_phi(torch.cat((x,y) , dim = 1))
    p = PINN_p(torch.cat((x,y) , dim = 1))
    T = PINN_T(torch.cat((x,y) , dim = 1))
    
    # Calculate gradients
    phi_y = torch.autograd.grad(phi, y, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    phi_x = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]

    u = phi_y  # u = ∂φ/∂y
    v = -phi_x # v = -∂φ/∂x

    # Calculate gradients
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]


    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    omega = v_x - u_y #vortisity scalar

    omega_x = torch.autograd.grad(omega, x, grad_outputs=torch.ones_like(omega), create_graph=True)[0]
    omega_y = torch.autograd.grad(omega, y, grad_outputs=torch.ones_like(omega), create_graph=True)[0]

    omega_xx = torch.autograd.grad(omega_x, x, grad_outputs=torch.ones_like(omega_x), create_graph=True)[0]
    omega_yy = torch.autograd.grad(omega_y, y, grad_outputs=torch.ones_like(omega_y), create_graph=True)[0]
    


    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
    T_yy = torch.autograd.grad(T_x, y, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
    
    
    # Continuity equation
    continuity_residual = u_x + v_y

    # Momentum equations
    nu = 0.01 #kinematic viscosity
    alpha = 0.02	#Diffusivity
    
    # Compute convective term: u ∂ω/∂x + v ∂ω/∂y   -  Diffusive term: ν (∂²ω/∂x² + ∂²ω/∂y²)
    omega_residual = (u * omega_x + v * omega_y ) - nu * (omega_xx + omega_yy)
     # Energy Equation
    energy_residual = u * T_x + v * T_y  - alpha * (T_xx + T_yy ) #- alpha_t_diff
    loss_mse = nn.MSELoss()
    #Note our target is zero. It is residual so we use zeros_like
    loss_pde = loss_mse(continuity_residual,torch.zeros_like(continuity_residual)) + loss_mse(omega_residual,torch.zeros_like(omega_residual)) + loss_mse(energy_residual,torch.zeros_like(energy_residual))
    return loss_pde


def add_gaussian_noise(tensor, mean=0.0, std_dev=0.01):
    noise = torch.normal(mean, std_dev, size=tensor.shape)
    return tensor + noise 


def boundary_condition_loss(PINN_u , PINN_v , PINN_p ,PINN_T , X , Y):
    
    u_b_pred = PINN_u(X)
    v_b_pred = PINN_v(X)
    p_b_pred = PINN_p(X)
    T_b_pred = PINN_T(X)
    
    loss_mse = nn.MSELoss()
    loss_u_b = loss_mse(u_b_pred  , (Y[: ,0]).reshape(-1 , 1))
    loss_v_b = loss_mse(v_b_pred  , (Y[: ,1]).reshape(-1 , 1))
    loss_p_b = loss_mse(p_b_pred  , (Y[: ,3]).reshape(-1 , 1))
    loss_T_b = loss_mse(T_b_pred  , (Y[: ,2]).reshape(-1 , 1))
    
    loss_bc = loss_u_b + loss_v_b + loss_p_b + loss_T_b
    return loss_bc


def data_loss(PINN_u , PINN_v , PINN_p ,PINN_T , X , Y):
    
    u_d_pred = PINN_u(X)
    v_d_pred = PINN_v(X)
    p_d_pred = PINN_p(X)
    T_d_pred = PINN_T(X)
    
    loss_mse = nn.MSELoss() 
    loss_u_d = loss_mse(u_d_pred  , (Y[: ,0].reshape(-1 , 1)))
    loss_v_d = loss_mse(v_d_pred  , (Y[: ,1].reshape(-1 , 1)))
    loss_p_d = loss_mse(p_d_pred  , (Y[: ,3].reshape(-1 , 1)))
    loss_T_d = loss_mse(T_d_pred  , (Y[: ,2].reshape(-1 , 1)))
    
    loss_data = loss_u_d + loss_v_d + loss_p_d + loss_T_d
    return loss_data

def noisy_data_loss(PINN_u , PINN_v , PINN_p ,PINN_T , X , Y):
    
    x_d_noisy = add_gaussian_noise(X[: , 0].reshape(-1 , 1))
    y_d_noisy = add_gaussian_noise(X[: , 1].reshape(-1 , 1))
    u_d_noisy = add_gaussian_noise(Y[: , 0].reshape(-1 , 1))
    v_d_noisy = add_gaussian_noise(Y[: , 1].reshape(-1 , 1))
    p_d_noisy = add_gaussian_noise(Y[: , 3].reshape(-1 , 1))
    T_d_noisy = add_gaussian_noise(Y[: , 2].reshape(-1 , 1))
    
    u_d_noisy_pred = PINN_u(X)
    v_d_noisy_pred = PINN_v(X)
    p_d_noisy_pred = PINN_p(X)
    T_d_noisy_pred = PINN_T(X)
    
    loss_mse = nn.MSELoss()
    loss_u_d_noisy = loss_mse(u_d_noisy_pred  , u_d_noisy)
    loss_v_d_noisy = loss_mse(v_d_noisy_pred  , v_d_noisy)
    loss_p_d_noisy = loss_mse(p_d_noisy_pred  , p_d_noisy)
    loss_T_d_noisy = loss_mse(T_d_noisy_pred  , T_d_noisy)
    
    loss_noisy_data = loss_u_d_noisy + loss_v_d_noisy + loss_p_d_noisy + loss_T_d_noisy
    return loss_noisy_data

def total_loss(PINN_u , PINN_v , PINN_p ,PINN_T , x_c , y_c,
               X_b , Y_b , X_d , Y_d):
    pde_loss = residula_loss(PINN_phi , PINN_p , PINN_T , x_c , y_c )
    bc_loss = boundary_condition_loss(PINN_u , PINN_v , PINN_p ,PINN_T , X_b , Y_b)
    interior_loss = data_loss(PINN_u , PINN_v , PINN_p ,PINN_T , X_d , Y_d)
    noisy_loss = noisy_data_loss(PINN_u , PINN_v , PINN_p ,PINN_T , X_d , Y_d)
    loss = (pde_loss * lambda_pde) + (bc_loss * lambda_bc) + (interior_loss * lambda_interior) + (noisy_loss * w_noise)
    return loss

def plot_results(PINN_u , PINN_v , PINN_T , PINN_p, file_tset):
    PINN_u.eval()
    PINN_v.eval()
    PINN_T.eval()
    PINN_p.eval()
    
    #df = normal_inputs(pd.read_csv(file_tset))
    df = pd.read_csv(file_tset)
    df = 2 * ((df - df.min()) / (df.max() - df.min())) - 1
    X = torch.tensor(df[['x','y']].values, dtype=torch.float32)
    Y = torch.tensor(df[['u' ,'v' ,'T' ,'p']].values, dtype=torch.float32)

    u_pred = PINN_u(X)
    v_pred = PINN_v(X)
    T_pred = PINN_T(X)
    p_pred = PINN_p(X)
    fig, ax= plt.subplots(nrows=2 , ncols=2 , figsize=(14, 10) , sharex = True)

                
    ax[0 , 0].set_title("u velocity")
    ax[0 , 0].plot(Y[: ,0] , label = "Exact")
    ax[0 , 0].plot(u_pred.detach().numpy() , label= "PINN")
    ax[0 , 0].legend(loc="upper right")
                
                
    ax[0 , 1].set_title("v velocity")
    ax[0 , 1].plot(Y[: ,1] , label = "Exact")
    ax[0 , 1].plot(v_pred.detach().numpy() , label= "PINN")
    ax[0 , 1].legend(loc="upper right")
                
    ax[1 , 1].set_title("T velocity")
    ax[1 , 1].plot(Y[: ,2] , label = "Exact")
    ax[1 , 1].plot(T_pred.detach().numpy() , label= "PINN")
    ax[1 , 1].legend(loc="upper right")
                
    ax[1 , 0].set_title("p velocity")
    ax[1 , 0].plot(Y[: ,3] , label = "Exact")
    ax[1 , 0].plot(p_pred.detach().numpy() , label= "PINN")
    ax[1 , 0].legend(loc="upper right")
                
                
    fig.suptitle(f'Comparison With Unseen Data : {file_tset}')
    #fig.savefig("Results/Comparison_plot" + time.strftime("%Y-%m-%d %H%M%S") + ".png")
    plt.show()
    
###################################################################


# Dynamic learning rate and optimizer switch
def train_pinn(PINN_u , PINN_v , PINN_p ,PINN_T, filename_data, filename_bc  ,epochs_adam , epoch_lbgfs):  
    print("Train starting . . . ")
    # Simulate a long process  
    for i in tqdm(range(100)):  
        time.sleep(0.05)  # Simulating work by sleeping
    tic = time.time()
        
    
    tic = time.time()
    optimizer_adam_u = optim.Adam(PINN_u.parameters(), lr=0.01)
    optimizer_adam_v = optim.Adam(PINN_v.parameters(), lr=0.01)
    optimizer_adam_T = optim.Adam(PINN_T.parameters(), lr=0.01)
    optimizer_adam_p = optim.Adam(PINN_p.parameters(), lr=0.01)
    
    scheduler_u = ReduceLROnPlateau(optimizer_adam_u , factor = 0.5 , min_lr = 1e-2 , verbose=False )
    scheduler_v = ReduceLROnPlateau(optimizer_adam_v , factor = 0.5 , min_lr = 1e-2 , verbose=False )
    scheduler_T = ReduceLROnPlateau(optimizer_adam_T , factor = 0.5 , min_lr = 1e-2 , verbose=False )
    scheduler_p = ReduceLROnPlateau(optimizer_adam_p , factor = 0.5 , min_lr = 1e-2 , verbose=False )
    
    
    # Define domain boundaries
    ub = torch.tensor([ 10, 5])
    lb = torch.tensor([ 0 , 0])


    # Number of points in each dimension
    n_points = 500  #collocation points number


    # Random sampling points in the domain
    x = np.random.uniform(lb[0], ub[0], n_points)
    y = np.random.uniform(lb[1], ub[1], n_points)
    np.random.seed(50)


    # Convert to PyTorch tensors for use in the PINN model
    X_c = torch.tensor(x, dtype=torch.float32, requires_grad=True).view(-1, 1)
    Y_c = torch.tensor(y, dtype=torch.float32, requires_grad=True).view(-1, 1)
    
    
    interior_data = normal_inputs(pd.read_csv(filename_data))
    boundary_data = normal_inputs(pd.read_csv(filename_bc))
    x_c = normal_inputs(X_c)
    y_c = normal_inputs(Y_c)
    XY_c = torch.cat((x_c , y_c) , dim = 1)

    X_interior = torch.tensor(interior_data[['x','y']].values, dtype=torch.float32)
    Y_interior = torch.tensor(interior_data[['u' ,'v' ,'T' ,'p']].values, dtype=torch.float32)
    X_boundary = torch.tensor(boundary_data[['x','y']].values, dtype=torch.float32)
    Y_boundary = torch.tensor(boundary_data[['u','v','T' ,'p']].values, dtype=torch.float32)
    
 
    
                
    # Hyperparameter tuning with Optuna
    def objective(trial):
            
        # Hyperparameters for tuning
        lambda_pde = trial.suggest_int("lambda_pde", 1, 10)
        lambda_interior = trial.suggest_int("lambda_interior", 0 ,10)
        lambda_bc = trial.suggest_int("lambda_bc", 0, 10)
                
            
        opt_u_adam = optim.Adam(PINN_u.parameters() , lr = 1e-2)
        opt_v_adam = optim.Adam(PINN_v.parameters() , lr = 1e-2)
        opt_p_adam = optim.Adam(PINN_u.parameters() , lr = 1e-2)
        opt_T_adam = optim.Adam(PINN_T.parameters() , lr = 1e-2)
        
        num_epochs = 50  #best value is 500
        for epoch in range(num_epochs):
            opt_u_adam.zero_grad()
            opt_v_adam.zero_grad()
            opt_p_adam.zero_grad()
            opt_T_adam.zero_grad()
            loss = total_loss(PINN_u , PINN_v , PINN_p ,PINN_T , x_c , y_c,
                              X_boundary , Y_boundary , X_interior , Y_interior)

            loss.backward(retain_graph=True)
            opt_u_adam.step()
            opt_v_adam.step()
            opt_p_adam.step()
            opt_T_adam.step()

            # Return the final loss for this trial
            return loss.item()
                
    num_trials= 5
    # Run the Optuna hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials = num_trials)  # Adjust n_trials for more thorough search

    # Extract the best lambda values
    best_params = study.best_params
    print("Optimized lambda_pde:", best_params["lambda_pde"])
    print("Optimized lambda_interior:", best_params["lambda_interior"])
    print("optimized lambda_bc:", best_params["lambda_bc"])

    lambda_pde = best_params["lambda_pde"]
    lambda_interior = best_params["lambda_interior"]
    lambda_bc = best_params["lambda_bc"]
    
    fold_loss = []  # Define fold_loss here to store the validation loss of each fold
    k = 4
    kf = KFold(n_splits=k)
    
    #for fold, (train_index, val_index) in enumerate(kf.split(X_interior)):
    for ((train_index, val_index) ,(train_c_index , val_c_index))  in zip(kf.split(X_interior) , kf.split(XY_c)):
        
        train_dataset = TensorDataset(X_interior[train_index], Y_interior[train_index])
        val_dataset = TensorDataset(X_interior[val_index], Y_interior[val_index])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        #collocation loader
        train_c_dataset = TensorDataset(XY_c[train_index])
        val_c_dataset = TensorDataset(XY_c[val_index])
        train_c_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_c_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        #print(f"Fold {fold + 1}/{k}")  
    
        for epoch in range(epochs_adam):
            PINN_u.train()
            PINN_v.train()
            PINN_T.train()
            PINN_p.train()
                
            t_loss = 0
            tic_batch = time.time()
            for batch ,batch_c in zip(train_loader , train_c_loader):
                
                X_data = batch[0]
                Y_data = batch[1]
                x_col = batch_c[0]
                y_col = batch_c[1]
                # Compute predictions and loss
                    
                loss  = total_loss(PINN_u , PINN_v , PINN_p ,PINN_T , x_col , y_col,
                                   X_boundary , Y_boundary , X_data , Y_data)
                t_loss += loss.item()
                    
                # Backpropagation
                optimizer_adam_u.zero_grad()
                optimizer_adam_v.zero_grad()
                optimizer_adam_T.zero_grad()
                optimizer_adam_p.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_adam_u.step()
                optimizer_adam_v.step()
                optimizer_adam_T.step()
                optimizer_adam_p.step()
                
            toc_batch = time.time()
            duration_time = toc_batch - tic_batch
            print(f"Batch_Duration :{duration_time :.2f} s")
            
            # Compute average training loss for the epoch
            avg_train_loss = t_loss / len(train_loader)
            # Validation and learning rate scheduling
            PINN_u.eval()
            PINN_v.eval()
            PINN_T.eval()
            PINN_p.eval()
            t_val_loss = 0
            #with torch.no_grad():
            loss  = total_loss(PINN_u , PINN_v , PINN_p ,PINN_T , x_c , y_c,
                                    X_boundary , Y_boundary , X_interior , Y_interior)
                    
            val_loss = sum(loss  for batch in val_loader) / len(val_loader)
            t_val_loss += val_loss.item()
            scheduler_u.step(val_loss)
            scheduler_v.step(val_loss)
            scheduler_T.step(val_loss)
            scheduler_p.step(val_loss)
            
            avg_val_loss = t_val_loss / len(val_loader)
            
            if epoch % 10 == 0:
                print(f'Epoch Adam {epoch}/{epochs_adam} [Fold:{fold}] [{100 * epoch/epochs_adam :.2f}%]  ,Total Loss: {loss.item():.6f}  ')
                print(f"Average Train Loss: {avg_train_loss :.5f}  |  Average Val Loss : {avg_val_loss :.5f}")
                print(f"======================================================================")
    
        # Record average validation loss per fold
        fold_loss.append(avg_train_loss)
        fold_loss.append(avg_val_loss)
    
        
        
    # Compute overall cross-validation loss
    cross_val_loss = sum(fold_loss) / len(fold_loss)
    print(f"[Fold{fold / k }] , overall cross-validation loss: {cross_val_loss:.5f}")
            
                
                        
    toc = time.time()
    elapseTime = toc - tic
    print ("elapse time in parallel = ", str(round(elapseTime , 4)) + " s")
        
    



lambda_pde = 1 #12
lambda_interior = 1 #4
lambda_bc = 1 #0.05
w_noise = 0.00000010

epoch_lbgfs = 50
epochs_adam = 300

# Create train/validation datasets
# Load data from CSV files
filename_data = r'E:/FOAM_PINN/cavHeat/twoD_lamin_over_box/2D_newData.csv'
filename_bc =  r'E:/FOAM_PINN/cavHeat/twoD_lamin_over_box/BC_data_2D_Lamin.csv'
f_test = r"E:/FOAM_PINN/cavHeat/twoD_lamin_over_box/2D_newTest.csv"



train_pinn(PINN_u , PINN_v , PINN_p ,PINN_T, filename_data, filename_bc  ,epochs_adam , epoch_lbgfs)


############################ plot funcs ################



plot_results(PINN_u , PINN_v , PINN_T , PINN_p, f_test )
    
    

        