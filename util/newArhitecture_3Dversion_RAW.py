import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import torch.optim as optim
import optuna
import time
import matplotlib.pyplot as plt

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
    

################################################################

#PINN_u is model(networks foe each components)
input_n = 3
h_n = 20
PINN_u = PINN_u().to(device)
PINN_v = PINN_v().to(device)
PINN_w = PINN_w().to(device)
PINN_p = PINN_p().to(device)		
PINN_T = PINN_T().to(device)

###################################################################

def pde_residuals(PINN_u , PINN_v , PINN_W ,  PINN_p , PINN_T , x , y  , z):
        
    u = PINN_u(torch.cat((x, y , z) , dim = 1))
    v = PINN_v(torch.cat((x, y , z) , dim = 1))
    w = PINN_w(torch.cat((x, y , z) , dim = 1))
    p = PINN_p(torch.cat((x, y , z) , dim = 1))
    T = PINN_T(torch.cat((x ,y , z) , dim = 1))
    
    # Calculate gradients
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True)[0]
     

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]


    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]
    
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]


    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_z = torch.autograd.grad(T, z, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
    T_yy = torch.autograd.grad(T_x, y, grad_outputs=torch.ones_like(T_y), create_graph=True)[0]
    T_zz = torch.autograd.grad(T_z, z, grad_outputs=torch.ones_like(T_z), create_graph=True)[0]
    
    
    # Continuity equation
    continuity_residual = u_x + v_y + w_z
    
    # Momentum equations
    nu = 0.01 #kinematic viscosity
    alpha = 0.02	#Diffusivity
    momentum_u_residual = (u * u_x + v * u_y  + w * u_z) + p_x - nu * (u_xx + u_yy + u_zz)
    momentum_v_residual = (u * v_x + v * v_y  + w * v_z) + p_y - nu * (v_xx + v_yy + v_zz)
    momentum_w_residual = (u * w_x + v * w_y  + w * w_z) + p_z - nu * (w_xx + w_yy + w_zz)
    
     # Energy Equation
    energy_residual = u * T_x + v * T_y  + w * T_z - alpha * (T_xx + T_yy + T_zz) #- alpha_t_diff
    loss_mse = nn.MSELoss()
    #Note our target is zero. It is residual so we use zeros_like
    loss_pde = loss_mse(continuity_residual,torch.zeros_like(continuity_residual)) + \
               loss_mse(momentum_u_residual,torch.zeros_like(momentum_u_residual)) + \
               loss_mse(momentum_v_residual,torch.zeros_like(momentum_v_residual)) + \
               loss_mse(momentum_w_residual,torch.zeros_like(momentum_w_residual))+ \
               loss_mse(energy_residual,torch.zeros_like(energy_residual))
    
    return loss_pde



def normal_inputs(df): #df is a dataframe
    normal_df = (df - df.min()) / (df.max() - df.min() )
    return normal_df

def rev_normal_results(out): #reverse normal data (out) to real scale
    rev_normal_res = (out * (out.max() - out.min())) + out.min()
    return rev_normal_res

def load_inputs_bc(filename_bc):
    data = pd.read_csv(filename_bc)
    n_data = normal_inputs(data)  
    n_data[['v']] = n_data[['w']] = 0		#check any features should revalue or not. in the case that after normalization be NaN
    n_data[['y']] = 0		#check any features should revalue or not. in the case that after normalization be NaN
    x = torch.tensor(n_data[['x']].values, dtype=torch.float32)
    y = torch.tensor(n_data[['y']].values, dtype=torch.float32)
    z = torch.tensor(n_data[['z']].values, dtype=torch.float32)
    u = torch.tensor(n_data[['u']].values, dtype=torch.float32)
    v = torch.tensor(n_data[['v']].values, dtype=torch.float32)
    w = torch.tensor(n_data[['w']].values, dtype=torch.float32)
    p = torch.tensor(n_data[['p']].values, dtype=torch.float32)
    T = torch.tensor(n_data[['T']].values, dtype=torch.float32)
    return x , y , z , u , v , w, p , T

def boundary_condition_loss(PINN_u , PINN_v , PINN_w ,  PINN_p ,PINN_T , filename_bc):
    x_b , y_b , z_b , u_b , v_b , w_b ,p_b , T_b = load_inputs_bc(filename_bc)
    u_b_pred = PINN_u(torch.cat((x_b , y_b  , z_b ) , dim = 1))
    v_b_pred = PINN_v(torch.cat((x_b , y_b  , z_b ) , dim = 1))
    w_b_pred = PINN_w(torch.cat((x_b , y_b  , z_b ) , dim = 1))
    p_b_pred = PINN_p(torch.cat((x_b , y_b  , z_b ) , dim = 1))
    T_b_pred = PINN_T(torch.cat((x_b , y_b  , z_b ) , dim = 1))
    
    loss_mse = nn.MSELoss()
    loss_u_b = loss_mse(u_b_pred  , u_b)
    loss_v_b = loss_mse(v_b_pred  , v_b)
    loss_w_b = loss_mse(w_b_pred  , w_b)
    loss_p_b = loss_mse(p_b_pred  , p_b)
    loss_T_b = loss_mse(T_b_pred  , T_b)
    
    loss_bc = loss_u_b + loss_v_b + loss_w_b +  loss_p_b + loss_T_b
    return loss_bc

def load_inputs_data(filename_data):
    data = pd.read_csv(filename_data)
    n_data = normal_inputs(data)  
    #n_data[['x']] = 0		#check any features should revalue or not. in the case that after normalization be NaN
    n_data[['z']] = 0		#check any features should revalue or not. in the case that after normalization be NaN
    x = torch.tensor(n_data[['x']].values, dtype=torch.float32)
    y = torch.tensor(n_data[['y']].values, dtype=torch.float32)
    z = torch.tensor(n_data[['z']].values, dtype=torch.float32)
    u = torch.tensor(n_data[['u']].values, dtype=torch.float32)
    v = torch.tensor(n_data[['v']].values, dtype=torch.float32)
    w = torch.tensor(n_data[['w']].values, dtype=torch.float32)
    p = torch.tensor(n_data[['p']].values, dtype=torch.float32)
    T = torch.tensor(n_data[['T']].values, dtype=torch.float32)
    return x , y , z , u , v , w , p , T

def data_loss(PINN_u , PINN_v , PINN_w , PINN_p ,PINN_T , filename_data):
    x_d , y_d , z_d , u_d , v_d , w_d , p_d , T_d = load_inputs_data(filename_data)
    u_d_pred = PINN_u(torch.cat((x_d , y_d , z_d) , dim = 1))
    v_d_pred = PINN_v(torch.cat((x_d , y_d , z_d) , dim = 1))
    w_d_pred = PINN_w(torch.cat((x_d , y_d , z_d) , dim = 1))
    p_d_pred = PINN_p(torch.cat((x_d , y_d  ,z_d) , dim = 1))
    T_d_pred = PINN_T(torch.cat((x_d , y_d , z_d) , dim = 1))
    
    loss_mse = nn.MSELoss() 
    loss_u_d = loss_mse(u_d_pred  , u_d)
    loss_v_d = loss_mse(v_d_pred  , v_d)
    loss_w_d = loss_mse(w_d_pred  , w_d)
    loss_p_d = loss_mse(p_d_pred  , p_d)
    loss_T_d = loss_mse(T_d_pred  , T_d)
    
    loss_data = loss_u_d + loss_v_d + loss_w_d + loss_p_d + loss_T_d
    return loss_data

def total_loss(PINN_u , PINN_v , PINN_w , PINN_p ,PINN_T , 
               filename_data , filename_bc ,
               x_c , y_c , z_c , lambda_pde , lambda_data , lambda_bc):
    calc_loss_data = data_loss(PINN_u , PINN_v , PINN_w , PINN_p ,PINN_T , filename_data)
    calc_loss_bc  = boundary_condition_loss(PINN_u , PINN_v , PINN_w ,  PINN_p ,PINN_T , filename_bc) 
    calc_loss_pde = pde_residuals(PINN_u , PINN_v , PINN_w , PINN_p , PINN_T , x_c , y_c , z_c )
                    
    
    
    loss = calc_loss_bc * lambda_bc + calc_loss_data * lambda_data + calc_loss_pde * lambda_pde
    return loss , calc_loss_data , calc_loss_bc , calc_loss_pde
 


    
#important define train section regarding multiple optimization and dynamic learning-rate
def train(PINN_u , PINN_v , PINN_w , PINN_p , PINN_T ,
          filename_data , filename_bc , 
          x_c , y_c , z_c , lambda_pde , lambda_data , lambda_bc):
    
    lambda_pde = 1.0
    lambda_data = 1.0
    lambda_bc = 1.0
    
    
    loss , calc_loss_data , calc_loss_bc , calc_loss_pde = total_loss(PINN_u , PINN_v , PINN_w ,  PINN_p ,PINN_T , 
               filename_data , filename_bc ,
               x_c , y_c , z_c , lambda_pde , lambda_data , lambda_bc)
    
    def objective(trial):
        
    
    # Hyperparameters for tuning
        lambda_pde = trial.suggest_float("lambda_pde", 0.1, 10.0)
        lambda_data = trial.suggest_float("lambda_data", 0.1, 10.0)
        lambda_bc = trial.suggest_float("lambda_bc", 0.1, 10.0)
        
    
        opt_u_adam = optim.Adam(PINN_u.parameters() , lr = 1e-3)
        opt_v_adam = optim.Adam(PINN_v.parameters() , lr = 1e-3)
        opt_w_adam = optim.Adam(PINN_w.parameters() , lr = 1e-3)
        opt_p_adam = optim.Adam(PINN_u.parameters() , lr = 1e-3)
        opt_T_adam = optim.Adam(PINN_T.parameters() , lr = 1e-3)
        
        num_epochs = 100
        for epoch in range(num_epochs):
            opt_u_adam.zero_grad()
            opt_v_adam.zero_grad()
            opt_w_adam.zero_grad()
            opt_p_adam.zero_grad()
            opt_T_adam.zero_grad()
            loss , calc_loss_data , calc_loss_bc , calc_loss_pde = total_loss(PINN_u , PINN_v , PINN_w , PINN_p ,PINN_T , 
                filename_data , filename_bc ,
                x_c , y_c , z_c , lambda_pde , lambda_data , lambda_bc)

            loss.backward()
            opt_u_adam.step()
            opt_v_adam.step()
            opt_w_adam.step()
            opt_p_adam.step()
            opt_T_adam.step()

        # Return the final loss for this trial
            return loss.item()
        
    num_trials= 100
    # Run the Optuna hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials = num_trials)  # Adjust n_trials for more thorough search

    # Extract the best lambda values
    best_params = study.best_params
    print("Optimized lambda_pde:", best_params["lambda_pde"])
    print("Optimized lambda_data:", best_params["lambda_data"])
    print("Optimized lambda_BC:", best_params["lambda_bc"])

    lambda_pde = best_params["lambda_pde"]
    lambda_data = best_params["lambda_data"]
    lambda_bc = best_params["lambda_bc"]
        
    opt_u_adam = optim.Adam(PINN_u.parameters() , lr = 1e-3)
    opt_v_adam = optim.Adam(PINN_v.parameters() , lr = 1e-3)	
    opt_w_adam = optim.Adam(PINN_w.parameters() , lr = 1e-3)	
    opt_p_adam = optim.Adam(PINN_p.parameters() , lr = 1e-3)	
    opt_T_adam = optim.Adam(PINN_T.parameters() , lr = 1e-3)	
    
      
    
    scheduler_u = ReduceLROnPlateau(opt_u_adam , factor = 0.5 , min_lr = 5e-5 , verbose=True )
    scheduler_v = ReduceLROnPlateau(opt_v_adam , factor = 0.5 , min_lr = 5e-5 , verbose=True )
    scheduler_w = ReduceLROnPlateau(opt_w_adam , factor = 0.5 , min_lr = 5e-5 , verbose=True )
    scheduler_p = ReduceLROnPlateau(opt_p_adam , factor = 0.5 , min_lr = 5e-5 , verbose=True )
    scheduler_T = ReduceLROnPlateau(opt_T_adam , factor = 0.5 , min_lr = 5e-5 , verbose=True )
    
    opt_u_lbfgs=torch.optim.LBFGS(PINN_u.parameters(),
      lr=0.1,  # or adjust based on your problem
      max_iter=100,  # More iterations for better convergence
      max_eval=None,  # Default
      tolerance_grad=1e-7,  # Increase sensitivity to gradients
      tolerance_change=1e-9,  # Keep default unless facing early stops
      history_size=100,  # Use larger history for better approximations
      line_search_fn="strong_wolfe")  # Use strong Wolfe line search for better convergence


    
    opt_v_lbfgs=torch.optim.LBFGS(PINN_v.parameters(),
      lr=0.1,  # or adjust based on your problem
      max_iter=100,  # More iterations for better convergence
      max_eval=None,  # Default
      tolerance_grad=1e-7,  # Increase sensitivity to gradients
      tolerance_change=1e-9,  # Keep default unless facing early stops
      history_size=100,  # Use larger history for better approximations
      line_search_fn="strong_wolfe")  # Use strong Wolfe line search for better convergence
    
    opt_p_lbfgs=torch.optim.LBFGS(PINN_p.parameters(),
      lr=0.1,  # or adjust based on your problem
      max_iter=100,  # More iterations for better convergence
      max_eval=None,  # Default
      tolerance_grad=1e-7,  # Increase sensitivity to gradients
      tolerance_change=1e-9,  # Keep default unless facing early stops
      history_size=100,  # Use larger history for better approximations
      line_search_fn="strong_wolfe")  # Use strong Wolfe line search for better convergence
    
    opt_T_lbfgs=torch.optim.LBFGS(PINN_T.parameters(),
      lr=0.1,  # or adjust based on your problem
      max_iter=100,  # More iterations for better convergence
      max_eval=None,  # Default
      tolerance_grad=1e-7,  # Increase sensitivity to gradients
      tolerance_change=1e-9,  # Keep default unless facing early stops
      history_size=100,  # Use larger history for better approximations
      line_search_fn="strong_wolfe")  # Use strong Wolfe line search for better convergence
    
    tic = time.time()
       
    
    def closure():
        opt_u_lbfgs.zero_grad()
        opt_v_lbfgs.zero_grad()
        opt_p_lbfgs.zero_grad()
        opt_T_lbfgs.zero_grad()
        
        
        loss , calc_loss_data , calc_loss_bc , calc_loss_pde = total_loss(PINN_u , PINN_v , PINN_w , PINN_p ,PINN_T , 
               filename_data , filename_bc ,
               x_c , y_c , z_c , lambda_pde , lambda_data , lambda_bc)
      
        loss.backward()
  
        return loss
    
    
    hist_val = []  # this is a list for storing loss values. this helps us to plot loss vs epochs
    
    #loop Adam
    
    for epo in range(epoch_adam):
        opt_u_adam.zero_grad()
        opt_v_adam.zero_grad()
        opt_w_adam.zero_grad()
        opt_p_adam.zero_grad()
        opt_T_adam.zero_grad()
        
        loss , calc_loss_data , calc_loss_bc , calc_loss_pde = total_loss(PINN_u , PINN_v , PINN_w , PINN_p ,PINN_T , 
               filename_data , filename_bc ,
               x_c , y_c , z_c , lambda_pde , lambda_data , lambda_bc)

        loss.backward()
        
        hist_val.append(loss.item())
        
        opt_u_adam.step()
        opt_v_adam.step()
        opt_w_adam.step()
        opt_p_adam.step()
        opt_T_adam.step()
        
        scheduler_u.step(loss)
        scheduler_v.step(loss)
        scheduler_w.step(loss)
        scheduler_p.step(loss)
        scheduler_T.step(loss)
        
        if epo %100 == 0:
            print(f'Epoch Adam {epo}/{epoch_adam}, Total Loss: {loss.item():.6f} , Learning Rate is: {scheduler_u.get_last_lr() + scheduler_v.get_last_lr() + scheduler_p.get_last_lr() + scheduler_T.get_last_lr()}')
            print(f'PDE Loss {calc_loss_pde:.4f}')
            print(f'Data loss {calc_loss_data:.4f}')
            print(f'Boundary condition loss {calc_loss_bc:.4f}')
            print("=====================================================================================================================")
        """    
        if loss.item() <=0.09 or loss.item() > 5:
            print("Optimization Method is switching to LBGF-S . . . ")
            break
        """
    
    """
    # loop LBFGS
    for epo in range(epoch_lbgfs):
        
        loss_u = opt_u_lbfgs.step(closure)
        loss_v = opt_u_lbfgs.step(closure)
        loss_p = opt_p_lbfgs.step(closure)
        loss_T = opt_T_lbfgs.step(closure)
        loss = (loss_u + loss_v + loss_p + loss_T) / 4.0
        hist_val.append(loss.item())
    
        if epo % 10 == 0:
            print(f'Epoch LBGF-s {epo}, Total Loss: {loss.item():.5f}')
    """
    
    
    #Directory for saving results
    path = r"E:\FOAM_PINN\cavHeat\CaseC\code\PINN_results"
    
    #plot residulas
    plt.figure(dpi = 100)
    plt.plot(hist_val , "b" ,label = "Loss value ")
    plt.title("loss valus VS Epochs")
    plt.xlabel("epochs")
    plt.ylabel("Loss values")
    plt.legend(loc = "best")
    plt.savefig(path + "loss_output_{epoch_adam}.png")
    plt.show()
    
    #elapsed time calculation
    
    toc = time.time()
    elapseTime = toc - tic
    print ("elapse time in parallel = ", str(round(elapseTime , 4)) + " s")
    ###################
    

    torch.save(PINN_u.state_dict(),path+"model_u" + ".pt")
    torch.save(PINN_v.state_dict(),path+"model_v" + ".pt")
    torch.save(PINN_w.state_dict(),path+"model_w" + ".pt")
    torch.save(PINN_p.state_dict(),path+"model_p" + ".pt")
    torch.save(PINN_T.state_dict(),path+"model_T" + ".pt")
    print ("Data saved!")
    
    
    #plotting predictions
        
    
    x_t , y_t , z_t , u_t , v_t , w_t , p_t , T_t = load_inputs_data(filename_data)
    
    plt.figure(figsize = (9 , 6) , dpi = 100)
    plt.subplot(2,1,1)
    plt.title("u-velocity")
    u_pred = PINN_u(torch.cat((x_t , y_t , z_t) , dim=1))
    plt.plot(u_pred.detach().numpy() , ls = "" , color = "k" , marker = "+" , alpha = 0.3 , label = "PINN")
    plt.plot(u_t.detach().numpy() , ls = "" , color = "r" , marker = "o"  , label = "ground-truth")
    plt.legend(loc = "best")
    
    plt.subplot(2,1,2)
    plt.title("Temperature")
    T_pred = PINN_T(torch.cat((x_t , y_t , z_t) , dim=1))
    plt.plot(T_pred.detach().numpy() , ls = "" , color = "k" , marker = "+" , alpha = 0.3 , label = "PINN")
    plt.plot(T_t.detach().numpy() , ls = "" , color = "r" , marker = "o"  , label = "ground-truth")
    plt.legend(loc = "best")
    print("Plots are loading . . . ")
    time.sleep(5)
    plt.savefig(path + "u_prediction_{epoch_adam}.png")
    plt.show()
    
    

    
filename_data = r"E:\FOAM_PINN\cavHeat\CaseC\code\case_C_Buoyant_fine_z0.csv"
filename_bc =  r"E:\FOAM_PINN\cavHeat\CaseC\code\CaseC_buoyant_BC.csv"


# Define domain boundaries
ub = torch.tensor([ 15, 7.7 , 7.5])
lb = torch.tensor([-7.5 , 0 , -7.5])


# Number of points in each dimension
n_points = 10000  #collocation points number


# Random sampling points in the domain
x = np.random.uniform(lb[0], ub[0], n_points)
y = np.random.uniform(lb[1], ub[1], n_points)
z = np.random.uniform(lb[2], ub[2], n_points)
np.random.seed(50)


# Convert to PyTorch tensors for use in the PINN model
x_c = torch.tensor(x, dtype=torch.float32, requires_grad=True).view(-1, 1)
y_c = torch.tensor(y, dtype=torch.float32, requires_grad=True).view(-1, 1)
z_c = torch.tensor(z, dtype=torch.float32, requires_grad=True).view(-1, 1)


lambda_pde = 1.0
lambda_data = 1.0
lambda_bc = 1.0

epoch_lbgfs = 20
epoch_adam = 5000


train(PINN_u , PINN_v , PINN_w , PINN_p , PINN_T ,
          filename_data , filename_bc , 
          x_c , y_c , z_c , lambda_pde , lambda_data , lambda_bc)
