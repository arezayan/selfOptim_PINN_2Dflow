
"""
Case E
@author: Amirreza
contact me : arezayan87@gmail.com
This case solves 2D flow over a Box.
flow is laminar, incompressible and non-isothermal

"""



import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import optuna
import seaborn as sb
import os
#import plotting

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the PINN model
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(nn.Tanh())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



# define function for calculation of nu_t
def calculate_nu_t(k, epsilon):
    C_mu = 0.09
    nu_t = C_mu * (k ** 2) / epsilon
    return nu_t

# Define the dimensionless form of governing equations
def pde_residuals(model,x,y):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    

    uvp = model(torch.cat((x, y), dim=1))
    u = uvp[: , 0:1]
    v = uvp[: , 1:2]
    p = uvp[: , 2:3]
    T = uvp[: , 3:4]
    #k = uvp[: , 4:5]
    #epsilon = uvp[: , 5:6]
    
    

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
    
    
    

    # Define dimensionless parameters (e.g., Prandtl number Pr, Grashof number Gr)
    Pr = 0.71  # Example Prandtl number for air
    Gr = 1e6   # Example Grashof number
    Re = 1e5   # Example Renolds number
    Ri = 1    # Example Richardson number
    ro= 1.225
    conduction_coeff = 0.02
    cp = 1.0061E+3
    Pe = conduction_coeff /(ro * cp)
    g = 9.81
    T_ref = 293.0  # Reference temperature in Kelvin
    beta = 1.0 / T_ref
    nu = 0.01
    alpha = 0.02
    
    T_correction = g * beta * (T - T_ref) #Buoyancy term contribution

    
    #Navier-Stokes equations with RANS
    #nu_t = calculate_nu_t(k, epsilon)
    #tau_x = 2 * nu_t * u_x
    #tau_y = 2 * nu_t * u_y
    #tau_z = nu_t * (u_z + w_x + v_y)
    
    momentum_u_residual = (u * u_x + v * u_y ) + p_x - nu * (u_xx + u_yy)# - tau_x - T_correction
    momentum_v_residual = (u * v_x + v * v_y ) + p_y - nu * (v_xx + v_yy)# - tau_y - T_correction
    
    Pr_t = 0.85  # Assume a turbulent Prandtl number for air
    #alpha_t = nu_t / Pr_t      

    # Continuity equation
    continuity_residual = u_x + v_y 
    
    """
    # Turbulent diffusion term (for energy)
    alpha_t_diff = alpha_t * (torch.autograd.grad(T_x, x, torch.ones_like(T_x), create_graph=True)[0] +
                              torch.autograd.grad(T_y, y, torch.ones_like(T_y), create_graph=True)[0] )
    """
    # Residual
    energy_residual = u * T_x + v * T_y  - alpha * (T_xx + T_yy ) #- alpha_t_diff
    
    continuity_residual = torch.mean(continuity_residual**2)
    momentum_u_residual = torch.mean(momentum_u_residual**2)
    momentum_v_residual = torch.mean(momentum_v_residual**2)
    energy_residual = torch.mean(energy_residual**2)

    return continuity_residual, momentum_u_residual, momentum_v_residual,energy_residual



# Load Boundary Condition Data from CSV
def load_boundary_conditions(filename):
    data_bc = pd.read_csv(filename)
    data_bc = (data_bc - data_bc.min()) / (data_bc.max() - data_bc.min())
    data_bc[['u']] = data_bc[['v']]  = 0

    x_bc = torch.tensor(data_bc[['x']].values, dtype=torch.float32)
    y_bc = torch.tensor(data_bc[['y']].values, dtype=torch.float32)
    
    u_bc = torch.tensor(data_bc[['u']].values, dtype=torch.float32)
    v_bc = torch.tensor(data_bc[['v']].values, dtype=torch.float32)
    p_bc = torch.tensor(data_bc[['p']].values, dtype=torch.float32)
    
    return x_bc, y_bc ,  u_bc , v_bc , p_bc

def load_data(filename):
    data = pd.read_csv(filename)
    data = (data - data.min()) / (data.max() - data.min())
    
    data[['v']] = 0  
    x_data = torch.tensor(data[['x']].values, dtype=torch.float32)
    y_data = torch.tensor(data[['y']].values, dtype=torch.float32)
    
    u_data = torch.tensor(data[['u']].values, dtype=torch.float32)
    v_data = torch.tensor(data[['v']].values, dtype=torch.float32)
    p_data = torch.tensor(data[['p']].values, dtype=torch.float32)
    
    return x_data, y_data  , u_data , v_data , p_data

# define loss data
def loss_data(x_data ,y_data , u_data , v_data , p_data , lb,ub,model):

    uvwpT_data_pred = model(torch.cat((x_data , y_data ) , dim = 1))

    loss_u_data = torch.mean((u_data - uvwpT_data_pred[:,0:1])**2)
    loss_v_data = torch.mean((u_data - uvwpT_data_pred[:,0:1])**2)
    loss_p_data = torch.mean((v_data - uvwpT_data_pred[:,1:2])**2)
    loss_T_data = torch.mean((p_data - uvwpT_data_pred[:,3:4])**2)
    

    return loss_u_data + loss_v_data  + loss_p_data

# define loss boundary condition
def loss_bc(x_bc, y_bc ,  u_bc , v_bc , p_bc ,lb,ub,model):
    uvwpT_BC_pred = model(torch.cat((x_bc , y_bc ) , dim = 1))
    loss_u_bc = torch.mean((u_bc - uvwpT_BC_pred[:,0:1])**2)
    loss_v_bc = torch.mean((v_bc - uvwpT_BC_pred[:,1:2])**2)
    loss_p_bc = torch.mean((p_bc - uvwpT_BC_pred[:,2:3])**2)
    


    return loss_u_bc + loss_v_bc + loss_p_bc 

# new train function
def train(model, file_data ,file_bc, x_tesnor ,y_tensor , lb, ub, epochs_adam , epochs_lbgfs):
  x_bc, y_bc , u_bc , v_bc , p_bc   = load_boundary_conditions(file_bc)
  x_data, y_data , u_data , v_data , p_data   = load_data(file_data)

  lambda_ns = 1.0
  lambda_continuty = 1.0
  lambda_data = 1.0
  lambda_bc = 1.0
  lambda_energy = 1.0
  


  def loss_func(model, x_tensor , y_tensor ,
                x_bc, y_bc , u_bc , v_bc , p_bc ,
                x_data ,y_data  , u_data , v_data , p_data , lb,ub ,lambda_ns , lambda_continuty , lambda_data , lambda_bc):

    # Compute Boundary Loss
    # Compute Boundary Loss
    boundary_loss = loss_bc(x_bc, y_bc ,  u_bc , v_bc ,  p_bc ,lb,ub,model)
    data_loss = loss_data(x_data ,y_data ,  u_data , v_data , p_data , lb,ub,model)

    # Compute PDE Residuals
    continuity_residual, momentum_u_residual, momentum_v_residual, energy_residual = pde_residuals(model, x_tensor , y_tensor )

        # Balance the loss terms
    total_loss = lambda_bc * boundary_loss + lambda_continuty * continuity_residual + lambda_ns*(momentum_u_residual + momentum_v_residual ) + lambda_data * data_loss + lambda_energy * energy_residual
    return total_loss ,continuity_residual, momentum_u_residual, momentum_v_residual,  boundary_loss , data_loss , energy_residual


  # Define the Optuna objective function for hyperparameter tuning
  def objective(trial):
      np.set_printoptions(precision=4)  # control precision

      # Hyperparameters for tuning
      lambda_ns = trial.suggest_float("lambda_ns", 0.1, 10.0)
      lambda_continuity = trial.suggest_float("lambda_continuity", 0.1, 10.0)
      lambda_data = trial.suggest_float("lambda_data", 0.1, 10.0)
      lambda_bc = trial.suggest_float("lambda_bc", 0.1, 10.0)
      lambda_energy = trial.suggest_float("lambda_energy", 0.1, 10.0)
    

      # Create the model and optimizer
      optimizer = optim.Adam(model.parameters(), lr=0.001)

      # Sample some input data (x, y, z) within the simplified domain
      #x = torch.rand(100, 3)  # For example purposes, using random points

      # Training loop for this trial
      num_epochs = 10
      for epoch in range(num_epochs):
          optimizer.zero_grad()
          loss,continuity_residual, momentum_u_residual, momentum_v_residual,  boundary_loss , data_loss , energy_residual = loss_func(model, x_tensor , y_tensor ,
                        x_bc, y_bc , u_bc , v_bc , p_bc ,
                        x_data ,y_data  , u_data , v_data , p_data , lb,ub ,lambda_ns , lambda_continuty , lambda_data , lambda_bc)

          loss.backward()
          optimizer.step()

      # Return the final loss for this trial
      return loss.item()

  # Run the Optuna hyperparameter optimization
  study = optuna.create_study(direction="minimize")
  study.optimize(objective, n_trials=20)  # Adjust n_trials for more thorough search

  # Extract the best lambda values
  best_params = study.best_params
  print("Optimized lambda_ns:", best_params["lambda_ns"])
  print("Optimized lambda_continuity:", best_params["lambda_continuity"])

  lambda_ns = best_params["lambda_ns"]
  lambda_continuty = best_params["lambda_continuity"]
  lambda_data = best_params["lambda_data"]
  lambda_bc = best_params["lambda_bc"]
  
  # Define optimizer for all networks
  #optimizer = optim.Adam(list(net_u.parameters()) + list(net_v.parameters()) + list(net_w.parameters()), lr=0.001)
  
    # Define the optimizer
  optimizer_adam = torch.optim.Adam(
      model.parameters(),         # Model parameters
      lr=1e-3)

  #scheduler = ReduceLROnPlateau(optimizer_adam, 'min', patience=10, factor=0.5, min_lr=1e-6)
  scheduler = ReduceLROnPlateau(optimizer_adam, mode='min', factor=0.5, patience=10, verbose=True)

  # Assuming you have three networks: u_net, v_net, w_net for velocity components
  #parameters = list(u_net.parameters()) + list(v_net.parameters()) + list(w_net.parameters())

  # Define the L-BFGS optimizer
  #optimizer = optim.LBFGS(parameters, lr=0.1, max_iter=500, history_size=10)

  optimizer_lbfgs=torch.optim.LBFGS(model.parameters(),
      lr=0.1,  # or adjust based on your problem
      max_iter=100,  # More iterations for better convergence
      max_eval=None,  # Default
      tolerance_grad=1e-7,  # Increase sensitivity to gradients
      tolerance_change=1e-9,  # Keep default unless facing early stops
      history_size=100,  # Use larger history for better approximations
      line_search_fn="strong_wolfe"  # Use strong Wolfe line search for better convergence

  )

  def closure():
    optimizer_lbfgs.zero_grad()

    loss ,continuity_residual, momentum_u_residual, momentum_v_residual,  boundary_loss , data_loss , energy_residual = loss_func(model, x_tensor , y_tensor ,
                  x_bc, y_bc , u_bc , v_bc , p_bc ,
                  x_data ,y_data  , u_data , v_data , p_data , lb,ub ,lambda_ns , lambda_continuty , lambda_data , lambda_bc)
    loss.backward()
    return loss
  hist=[]
  for epo in range(epochs_adam):
      model.train()
      optimizer_adam.zero_grad()
      loss ,continuity_residual, momentum_u_residual, momentum_v_residual,  boundary_loss , data_loss , energy_residual = loss_func(model, x_tensor , y_tensor ,
                    x_bc, y_bc , u_bc , v_bc , p_bc ,
                    x_data ,y_data  , u_data , v_data , p_data , lb,ub ,lambda_ns , lambda_continuty , lambda_data , lambda_bc)
      loss.backward()
      hist.append(loss.item())
      optimizer_adam.step()
      scheduler.step(loss)

      if epo %50 == 0:
        print(f'Epoch Adam {epo}/{epochs_adam}, Total Loss: {loss.item():.6f} , Learning Rate is: {scheduler.get_last_lr()}')
        print(f'Momentum Loss {torch.sum(momentum_u_residual**2 + momentum_v_residual**2 ):.4f}')
        print(f'Lambda data {lambda_data:.4f}')
        
        print(f'Data Loss {data_loss.item():.4f}')
        print("=====================================================================================================================")
        
      if loss.item() <=0.09 or loss.item() > 5:
        print("Optimzation Method is swtiching to LBGF-S . . . ")
        break

  for epochs in range(epochs_lbgfs):    
        model.train()
        loss = optimizer_lbfgs.step(closure)
        hist.append(loss.item())
    
        if epochs % 10 == 0:
            print(f'Epoch LBGF-s {epochs}, Total Loss: {loss.item():.5f}')

       #print(f'The highest Loss is:  {max(momentum_loss.item() , continuity_loss.item() , loss_data.item() , loss_bc.item()):.6f}')
       #print(time.time())
        """
      u = (model(torch.cat((x, y, z), dim=1)))[:,0:1].detach()
      plt.figure(figsize=(6,2.5))
      plt.scatter(z.detach().numpy(), u_exact.detach().numpy(),color = "k", label="Noisy observations", alpha=0.6)
      plt.plot(z.detach().numpy(), u.detach().numpy(),ls = "",marker = "+",  label="PINN solution", color="tab:green")
      plt.title(f"Training step {epochs}")
      plt.legend(loc = "best")
      #plt.savefig("/content/drive/MyDrive/cavity/singleCube/Results/output_{epochs}.png".format(epochs=epochs))
      #plt.show()
      
       """
  dirName = "Results"
  newpath = r'E:/FOAM_PINN/cavHeat/2D_lamin_over_box/' + dirName 
  if not os.path.exists(newpath):
    os.makedirs(newpath)  
  
  plt.figure(dpi = 100)
  plt.plot(range(len(hist)), hist, label="Training Loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.yscale("log")
  plt.title("Convergence Plot")
  plt.legend()
  plt.savefig( newpath + "/Convergence_{epochs}.png".format(epochs=epochs))
  plt.show()
  
  #plot _results
  uvwp = model(torch.cat((x_data, y_data), dim=1))
  predicted_u = uvwp[:,0]
    
  plt.figure(dpi = 300)
  plt.plot( predicted_u.detach().numpy() , ls = "" ,marker = "o", color = "tab:blue" , label = "PINN") 
  plt.plot( u_data.detach().numpy() ,  ls = "-" , marker = "+" ,color = "tab:green" , label = "ground truth")
  plt.xlabel("y : Height")
  plt.ylabel("u vel")
  plt.title("Grid of X and Y: over 2D Box case")
  plt.legend(loc = "best")

  plt.savefig( newpath + "/output_2D_over_Box_{epochs}.png".format(epochs=epochs))
  torch.save(model.state_dict(), 'E:/FOAM_PINN/cavHeat/2D_lamin_over_box/Results/PINN_case_2D_optuna.pth')
  
  
         
# Initialize and Train the Model
epochs_adam = 1500
epochs_lbgfs = 500

layers = [2, 50,50,50,50, 4]  # Customize the network layers
#layers_u = [2, 50,50,50,50, 1]  # Customize the network layers
#layers_v = [2, 50,50,50,50, 1]  # Customize the network layers
#layers_w = [2, 50,50,50,50, 1]  # Customize the network layers
#layers_p = [2, 50,50,50,50, 1]  # Customize the network layers


model = PINN(layers)
#model_u = PINN(layers_u)
#model_v = PINN(layers_v)
#model_w = PINN(layers_w)
#model_p = PINN(layers_p)

filename_data = r'sparce_data_2D_lamin.csv'
filename_bc =  r'BC_data_2D_Lamin.csv'


# Define domain boundaries
ub = torch.tensor([ 10, 5])
lb = torch.tensor([ 0 , 0])


# Number of points in each dimension
n_points = 10000  #collocation points number

# Random sampling points in the domain
x = np.random.uniform(lb[0], ub[0], n_points)
y = np.random.uniform(lb[1], ub[1], n_points)
np.random.seed(50)


# Convert to PyTorch tensors for use in the PINN model
x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32, requires_grad=True).view(-1, 1)


train(model,filename_data ,filename_bc, x_tensor ,y_tensor , lb, ub, epochs_adam , epochs_lbgfs)


