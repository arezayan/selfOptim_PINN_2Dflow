import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import torch.optim as optim
import optuna
import time
import matplotlib.pyplot as plt
from matplotlib import cm

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
h_n = 64
PINN_u = PINN_u().to(device)
PINN_v = PINN_v().to(device)
PINN_w = PINN_w().to(device)
PINN_p = PINN_p().to(device)
PINN_T = PINN_T().to(device)

PINN_u.apply(init_xavier)
PINN_v.apply(init_xavier)
PINN_p.apply(init_xavier)
PINN_T.apply(init_xavier)


###################################################################

def pde_residuals(PINN_u , PINN_v , PINN_p , PINN_T , x , y ):
    
    x = (2 * (x - x.min()) / (x.max() - x.min())) - 1
    y = (2 * (y - y.min()) / (y.max() - y.min())) - 1
        
    u = PINN_u(torch.cat((x,y) , dim = 1))
    v = PINN_v(torch.cat((x,y) , dim = 1))
    p = PINN_p(torch.cat((x,y) , dim = 1))
    T = PINN_T(torch.cat((x,y) , dim = 1))
    
    # Calculate gradients
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
     

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]


    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]


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
    momentum_u_residual = (u * u_x + v * u_y ) + p_x - nu * (u_xx + u_yy)
    momentum_v_residual = (u * v_x + v * v_y ) + p_y - nu * (v_xx + v_yy)
    
     # Energy Equation
    energy_residual = u * T_x + v * T_y  - alpha * (T_xx + T_yy ) #- alpha_t_diff
    loss_mse = nn.MSELoss()
    #Note our target is zero. It is residual so we use zeros_like
    loss_pde = loss_mse(continuity_residual,torch.zeros_like(continuity_residual)) + loss_mse(momentum_u_residual,torch.zeros_like(momentum_u_residual)) + loss_mse(momentum_v_residual,torch.zeros_like(momentum_v_residual)) + loss_mse(energy_residual,torch.zeros_like(energy_residual))
    return loss_pde



def normal_inputs(df): #df is a dataframe
    normal_df = (2 * (df - df.min()) / (df.max() - df.min() )) - 1
    return normal_df


# Function to inject Gaussian noise
def add_gaussian_noise(tensor, mean=0.0, std_dev=0.01):
    noise = torch.normal(mean, std_dev, size=tensor.shape)
    return tensor + noise 

def load_inputs_data(filename_data):
    data = pd.read_csv(filename_data)
    data = normal_inputs(data) # Normalize the data

    data_tensor = torch.tensor(data.values, dtype=torch.float32)
    

    x = data_tensor[: , 0].reshape(-1 , 1)
    y = data_tensor[: , 1].reshape(-1 , 1)
    u = data_tensor[: , 2].reshape(-1 , 1)
    v = data_tensor[: , 3].reshape(-1 , 1)
    p = data_tensor[: , 4].reshape(-1 , 1)
    T = data_tensor[: , 5].reshape(-1 , 1)

    return x , y , u , v , p , T



def boundary_condition_loss(PINN_u , PINN_v , PINN_p ,PINN_T , filename_bc):
    x_b , y_b , u_b , v_b , p_b , T_b = load_inputs_data(filename_bc)
    u_b_pred = PINN_u(torch.cat((x_b , y_b ) , dim = 1))
    v_b_pred = PINN_v(torch.cat((x_b , y_b ) , dim = 1))
    p_b_pred = PINN_p(torch.cat((x_b , y_b ) , dim = 1))
    T_b_pred = PINN_T(torch.cat((x_b , y_b ) , dim = 1))
    
    loss_mse = nn.MSELoss()
    loss_u_b = loss_mse(u_b_pred  , u_b)
    loss_v_b = loss_mse(v_b_pred  , v_b)
    loss_p_b = loss_mse(p_b_pred  , p_b)
    loss_T_b = loss_mse(T_b_pred  , T_b)
    
    loss_bc = loss_u_b + loss_v_b + loss_p_b + loss_T_b
    return loss_bc




def data_loss(PINN_u , PINN_v , PINN_p ,PINN_T , filename_data):
    x_d , y_d , u_d , v_d , p_d , T_d = load_inputs_data(filename_data)
    u_d_pred = PINN_u(torch.cat((x_d , y_d ) , dim = 1))
    v_d_pred = PINN_v(torch.cat((x_d , y_d ) , dim = 1))
    p_d_pred = PINN_p(torch.cat((x_d , y_d ) , dim = 1))
    T_d_pred = PINN_T(torch.cat((x_d , y_d ) , dim = 1))
    
    loss_mse = nn.MSELoss() 
    loss_u_d = loss_mse(u_d_pred  , u_d)
    loss_v_d = loss_mse(v_d_pred  , v_d)
    loss_p_d = loss_mse(p_d_pred  , p_d)
    loss_T_d = loss_mse(T_d_pred  , T_d)
    
    loss_data = loss_u_d + loss_v_d + loss_p_d + loss_T_d
    return loss_data

def noisy_data_loss(PINN_u , PINN_v , PINN_p ,PINN_T , filename_data):
    x_d , y_d , u_d , v_d , p_d , T_d = load_inputs_data(filename_data)
    x_d_noisy = add_gaussian_noise(x_d)
    y_d_noisy = add_gaussian_noise(y_d)
    u_d_noisy = add_gaussian_noise(u_d)
    v_d_noisy = add_gaussian_noise(v_d)
    p_d_noisy = add_gaussian_noise(p_d)
    T_d_noisy = add_gaussian_noise(T_d)
    
    u_d_noisy_pred = PINN_u(torch.cat((x_d_noisy , y_d_noisy ) , dim = 1))
    v_d_noisy_pred = PINN_v(torch.cat((x_d_noisy , y_d_noisy ) , dim = 1))
    p_d_noisy_pred = PINN_p(torch.cat((x_d_noisy , y_d_noisy ) , dim = 1))
    T_d_noisy_pred = PINN_T(torch.cat((x_d_noisy , y_d_noisy ) , dim = 1))
    
    loss_mse = nn.MSELoss()
    loss_u_d_noisy = loss_mse(u_d_noisy_pred  , u_d_noisy)
    loss_v_d_noisy = loss_mse(v_d_noisy_pred  , v_d_noisy)
    loss_p_d_noisy = loss_mse(p_d_noisy_pred  , p_d_noisy)
    loss_T_d_noisy = loss_mse(T_d_noisy_pred  , T_d_noisy)
    
    loss_noisy_data = loss_u_d_noisy + loss_v_d_noisy + loss_p_d_noisy + loss_T_d_noisy
    return loss_noisy_data
    
    
    

def total_loss(PINN_u , PINN_v , PINN_p ,PINN_T , 
               filename_data , filename_bc ,
               x_c , y_c , lambda_pde , lambda_data , lambda_bc):
    calc_loss_data = data_loss(PINN_u , PINN_v , PINN_p ,PINN_T , filename_data)
    calc_loss_noisy_data = noisy_data_loss(PINN_u , PINN_v , PINN_p , PINN_T , filename_data)
    calc_loss_bc  = boundary_condition_loss(PINN_u , PINN_v , PINN_p ,PINN_T , filename_bc) 
    calc_loss_pde = pde_residuals(PINN_u , PINN_v , PINN_p , PINN_T , x_c , y_c )
    
    loss = calc_loss_bc * lambda_bc + calc_loss_data * lambda_data + calc_loss_pde * lambda_pde + calc_loss_noisy_data * 10
    return loss , calc_loss_data * lambda_data , calc_loss_bc * lambda_bc , calc_loss_pde * lambda_pde  , calc_loss_noisy_data * 10



def sub_plot(fname , PINN_u , PINN_v , PINN_p , PINN_T , epo , title):

  x_d , y_d , u_d , v_d  , p_d , T_d = load_inputs_data(fname )
  #z_d = torch.tensor([0.0]*len(z_d), dtype=torch.float32).reshape(-1 ,1)  # dummy target for loss calculation, it should be real data

            
  u_pred = PINN_u( torch.cat((x_d , y_d ) , dim=1))
  v_pred = PINN_v( torch.cat((x_d , y_d ) , dim=1))
  T_pred = PINN_T( torch.cat((x_d , y_d ) , dim=1))
  p_pred = PINN_p( torch.cat((x_d , y_d ) , dim=1))
              
  fig , ax =plt.subplots(ncols=2 , nrows=2 , sharex = False , sharey = False)
  mse = nn.MSELoss()
  erroru = torch.mean(torch.abs(u_pred - u_d) / (torch.abs(u_d)+ 1e-15))

  ax[0 , 0].set_title('u velocity')
  ax[0 , 0].plot(u_d , label = "Exact")
  ax[0 , 0].plot(u_pred.detach().numpy() , label= "PINN" , alpha = 0.5)
  ax[0 , 0].legend(loc="upper right")
              
  errorv = torch.mean(torch.abs(v_pred - v_d) / (torch.abs(v_d)+ 1e-15))

  ax[0 , 1].set_title("v velocity")
  ax[0 , 1].plot(v_d , label = "Exact")
  ax[0 , 1].plot(v_pred.detach().numpy() , label= "PINN" , alpha = 0.5)
  ax[0 , 1].legend(loc="upper right")
              
  errorT = torch.mean(torch.abs(T_pred - T_d) / (torch.abs(T_d)+ 1e-15))
  ax[1 , 1].set_title("T ")
  ax[1 , 1].plot(T_d , label = "Exact")
  ax[1 , 1].plot(T_pred.detach().numpy() , label= "PINN", alpha = 0.5)
  ax[1 , 1].legend(loc="upper right")
              
  ax[1 , 0].set_title("p ")
  ax[1 , 0].plot(p_d , label = "Exact")
  ax[1 , 0].plot(p_pred.detach().numpy() , label= "PINN" , alpha = 0.5)
  ax[1 , 0].legend(loc="upper right") 
  
  fig.suptitle(f'Comparison {title} {epo}' )
  fig.savefig("Results/comaprision_"  + fname + ".png")
  print(f" {title}  Error u : {erroru:.3f}  | Error v : {errorv:.3f}  | Error T : {errorT:.3f}  |  ")
  #plt.show()

def simple_plots(fname , PINN_u , PINN_v , PINN_T ,  PINN_p , epo , title ):
    
    x_d , y_d ,  u_d , v_d  , T_d , p_d = load_inputs_data(fname)
    
    u_pred = PINN_u( torch.cat((x_d , y_d  ) , dim=1))
    v_pred = PINN_v( torch.cat((x_d , y_d  ) , dim=1))
    T_pred = PINN_T( torch.cat((x_d , y_d  ) , dim=1))
    p_pred = PINN_p( torch.cat((x_d , y_d  ) , dim=1))
  
              
    plt.figure(dpi = 100)
    plt.plot(u_d , label = "Exact")
    plt.plot(u_pred.detach().numpy() , label= "PINN" , alpha = 0.5)
    plt.legend(loc="best")
    plt.title(title + " Epoch " + str(epo))
    #plt.ylim(-1.0 , 1.0)
    plt.show()
    #z_d = torch.tensor([0.0]*len(z_d), dtype=torch.float32).reshape(-1 ,1)  # dummy target for loss calculation, it should be real data
 
  

    
#important define train section regarding multiple optimization and dynamic learning-rate
def train(PINN_u , PINN_v , PINN_p , PINN_T ,
          filename_data , filename_bc , 
          x_c , y_c , lambda_pde , lambda_data , lambda_bc):
    
    #lambda_pde = 1.0
    #lambda_data = 1.0
    #lambda_bc = 1.0
    
    
    loss , calc_loss_data , calc_loss_bc , calc_loss_pde , calc_loss_noisy_data = total_loss(PINN_u , PINN_v , PINN_p ,PINN_T , 
               filename_data , filename_bc ,
               x_c , y_c , lambda_pde , lambda_data , lambda_bc)
    
    def objective(trial):
        
    
    # Hyperparameters for tuning
        lambda_pde = trial.suggest_float("lambda_pde", 0.1, 10.0)
        lambda_data = trial.suggest_float("lambda_data", 1E-1, 10.0)
        lambda_bc = trial.suggest_float("lambda_bc", 1E-1, 10.0)
        
    
        opt_u_adam = optim.Adam(PINN_u.parameters() , lr = 1e-5)
        opt_v_adam = optim.Adam(PINN_v.parameters() , lr = 1e-5)
        opt_p_adam = optim.Adam(PINN_u.parameters() , lr = 1e-5)
        opt_T_adam = optim.Adam(PINN_T.parameters() , lr = 1e-5)
        
        num_epochs = 500  #best value is 500
        for epoch in range(num_epochs):
            opt_u_adam.zero_grad()
            opt_v_adam.zero_grad()
            opt_p_adam.zero_grad()
            opt_T_adam.zero_grad()
            loss , calc_loss_data , calc_loss_bc , calc_loss_pde , calc_loss_noisy_data = total_loss(PINN_u , PINN_v , PINN_p ,PINN_T , 
                filename_data , filename_bc ,
                x_c , y_c , lambda_pde , lambda_data , lambda_bc)

            loss.backward()
            opt_u_adam.step()
            opt_v_adam.step()
            opt_p_adam.step()
            opt_T_adam.step()

        # Return the final loss for this trial
            return loss.item()
        
    num_trials= 20
    # Run the Optuna hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials = num_trials)  # Adjust n_trials for more thorough search

    # Extract the best lambda values
    best_params = study.best_params
    print("Optimized lambda_pde:", best_params["lambda_pde"])
    print("Optimized lambda_data:", best_params["lambda_data"])
    print("optimized lambda_bc:", best_params["lambda_bc"])

    lambda_pde = best_params["lambda_pde"]
    lambda_data = best_params["lambda_data"]
    lambda_bc = best_params["lambda_bc"]
       
    opt_u_adam = optim.Adam(PINN_u.parameters() , lr = 1e-2)
    opt_v_adam = optim.Adam(PINN_v.parameters() , lr = 1e-2)
    opt_p_adam = optim.Adam(PINN_p.parameters() , lr = 1e-2)
    opt_T_adam = optim.Adam(PINN_T.parameters() , lr = 1e-2)



    scheduler_u = ReduceLROnPlateau(opt_u_adam , factor = 0.5 , min_lr = 1e-3 , verbose=False )
    scheduler_v = ReduceLROnPlateau(opt_v_adam , factor = 0.5 , min_lr = 1e-3 , verbose=False )
    scheduler_p = ReduceLROnPlateau(opt_p_adam , factor = 0.5 , min_lr = 1e-3 , verbose=False )
    scheduler_T = ReduceLROnPlateau(opt_T_adam , factor = 0.5 , min_lr = 1e-3 , verbose=False )
    
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
        
        
        loss , calc_loss_data , calc_loss_bc , calc_loss_pde , calc_loss_noisy_data = total_loss(PINN_u , PINN_v , PINN_p ,PINN_T , 
               filename_data , filename_bc ,
               x_c , y_c , lambda_pde , lambda_data , lambda_bc)
      
        loss.backward()
  
        return loss
    
    
    hist_val = []  # this isa list for storing loss values. this helps us to plot loss vs epochs
    
    #loop Adam
    
    for epo in range(epoch_adam):
        opt_u_adam.zero_grad()
        opt_v_adam.zero_grad()
        opt_p_adam.zero_grad()
        opt_T_adam.zero_grad()
        
        loss , calc_loss_data , calc_loss_bc , calc_loss_pde , calc_loss_noisy_data = total_loss(PINN_u , PINN_v , PINN_p ,PINN_T , 
               filename_data , filename_bc ,
               x_c , y_c , lambda_pde , lambda_data , lambda_bc)

        loss.backward()
        
        hist_val.append(loss.item())
        
        opt_u_adam.step()
        opt_v_adam.step()
        opt_p_adam.step()
        opt_T_adam.step()
        
        scheduler_u.step(loss)
        scheduler_v.step(loss)
        scheduler_p.step(loss)
        scheduler_T.step(loss)
        """
        if epo %200 == 0:
            print(f'Epoch Adam {epo}/{epoch_adam} [{100 * epo/epoch_adam :.2f}%]  ,Total Loss: {loss.item():.6f} , Learning Rate is: {scheduler_u.get_last_lr() + scheduler_v.get_last_lr() + scheduler_p.get_last_lr() + scheduler_T.get_last_lr()}')
            print(f'PDE Loss {calc_loss_pde:.4f} | Data loss {calc_loss_data:.4f} | Boundary condition loss {calc_loss_bc:.4f}')
            #print(f'Data loss {calc_loss_data:.4f}')
            #print(f'Boundary condition loss {calc_loss_bc:.4f}')
            print("=====================================================================================================================")
        """
        if epo % 200 == 0:
            print(f'Epoch Adam {epo}/{epoch_adam} [{100 * epo/epoch_adam :.2f}%]  ,Total Loss: {loss.item():.6f}  ')
            print(f'PDE Loss {calc_loss_pde:.4f} | Data loss {calc_loss_data:.4f} | Boundary condition loss {calc_loss_bc:.4f}')
            f_test_tempo = r"2D_newTest.csv"
            f_train_tempo = r'2D_newData.csv'
            
            
            sub_plot(f_train_tempo , PINN_u, PINN_v, PINN_p, PINN_T, epo , "train")
            print("=====================================================================================================================")
        if epo % 200 == 0:
            f_test_tempo = r"2D_newTest.csv"
            sub_plot(f_test_tempo , PINN_u, PINN_v, PINN_p, PINN_T, epo , "test")
            

            
        if loss.item() <=0.0005 :
            print("Optimization Method is switching to LBGF-S . . . ")
            break

    # loop LBFGS
    for epo in range(epoch_lbgfs):
        
        loss_u = opt_u_lbfgs.step(closure)
        loss_v = opt_u_lbfgs.step(closure)
        loss_p = opt_p_lbfgs.step(closure)
        loss_T = opt_T_lbfgs.step(closure)
        loss = (loss_u + loss_v + loss_p + loss_T) / 4.0
        hist_val.append(loss.item())
    
        if epo % 10 == 0:
            print(f'Epoch LBGF-S {epo}/{epoch_lbgfs} [{100 * epo/epoch_lbgfs :.2f}%]  ,Total Loss: {loss.item():.6f} , Learning Rate is: {scheduler_u.get_last_lr() + scheduler_v.get_last_lr() + scheduler_p.get_last_lr() + scheduler_T.get_last_lr()}')
            print(f'PDE Loss {calc_loss_pde:.4f} | Data loss {calc_loss_data:.4f} | Boundary condition loss {calc_loss_bc:.4f}')
    
    
    
    #Directory for saving results
    path = r"E:\FOAM_PINN\cavHeat\twoD_lamin_over_box\Results\PINN"
    
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
    torch.save(PINN_p.state_dict(),path+"model_p" + ".pt")
    torch.save(PINN_T.state_dict(),path+"model_T" + ".pt")
    print ("Data saved!")
    
    
    #plotting predictions
    #Final PLot
    plt.figure(dpi = 100)
    x_t , y_t , u_t , v_t , p_t , T_t = load_inputs_data(filename_data)
    u_pred = PINN_u(torch.cat((x_t , y_t) , dim=1))
    plt.title("U prediction")
    plt.plot(u_pred.detach().numpy() , ls = "" , color = "k" , marker = "+" , alpha = 0.3 , label = "PINN")
    plt.plot(u_t.detach().numpy() , ls = "" , color = "b" , marker = "o"  , label = "ground-truth")
    plt.xlabel("NONE")
    plt.ylabel("u Velocity")
    plt.legend(loc = "best")
    print("Plots are loading . . . ")
    time.sleep(2)
    plt.savefig(path + "u_prediction_{epoch_adam}.png")
    plt.show()
    
    

filename_data = r'2D_newData.csv'
filename_bc =  r'BC_data_2D_Lamin.csv'
f_test = r"2D_newTest.csv"


# Define domain boundaries
ub = torch.tensor([ 10, 5])
lb = torch.tensor([ 0 , 0])


# Number of points in each dimension
n_points = 1000  #collocation points number


# Random sampling points in the domain
x = np.random.uniform(lb[0], ub[0], n_points)
y = np.random.uniform(lb[1], ub[1], n_points)
np.random.seed(50)


# Convert to PyTorch tensors for use in the PINN model
x_c = torch.tensor(x, dtype=torch.float32, requires_grad=True).view(-1, 1)
y_c = torch.tensor(y, dtype=torch.float32, requires_grad=True).view(-1, 1)


lambda_pde = 1 #12
lambda_data = 1 #4
lambda_bc = 1 #0.05

epoch_lbgfs = 50
epoch_adam = 5000



train(PINN_u , PINN_v , PINN_p , PINN_T ,
          filename_data , filename_bc , 
          x_c , y_c , lambda_pde , lambda_data , lambda_bc)



x_d , y_d , u_d , v_d , p_d , T_d = load_inputs_data(f_test)
            
u_pred = PINN_u( torch.cat((x_d , y_d ) , dim=1))
v_pred = PINN_v( torch.cat((x_d , y_d ) , dim=1))
T_pred = PINN_T( torch.cat((x_d , y_d ) , dim=1))
p_pred = PINN_p( torch.cat((x_d , y_d ) , dim=1))
            
fig, ax= plt.subplots(nrows=2 , ncols=2 , figsize=(14, 10) , sharex = True)

            
ax[0 , 0].set_title("u velocity")
ax[0 , 0].plot(u_d , label = "Exact")
ax[0 , 0].plot(u_pred.detach().numpy() , label= "PINN")
ax[0 , 0].legend(loc="upper right")
            
            
ax[0 , 1].set_title("v velocity")
ax[0 , 1].plot(v_d , label = "Exact")
ax[0 , 1].plot(v_pred.detach().numpy() , label= "PINN")
ax[0 , 1].legend(loc="upper right")
            
ax[1 , 1].set_title("T velocity")
ax[1 , 1].plot(T_d , label = "Exact")
ax[1 , 1].plot(T_pred.detach().numpy() , label= "PINN")
ax[1 , 1].legend(loc="upper right")
            
ax[1 , 0].set_title("p velocity")
ax[1 , 0].plot(p_d , label = "Exact")
ax[1 , 0].plot(p_pred.detach().numpy() , label= "PINN")
ax[1 , 0].legend(loc="upper right")
            
            
fig.suptitle(f'Comparison With : {f_test}')
fig.savefig("Results/Comparison_plot" + time.strftime("%Y-%m-%d %H%M%S") + ".png")
plt.show()


