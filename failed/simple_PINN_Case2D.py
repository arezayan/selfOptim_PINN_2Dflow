
"""
Created for solving Stedy-State 2D inconpressible flow in a channel
Momentum and continuty eqyatuin is considered
x, y  : inputs
u,v,p : outputs
@author: Amirreza
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
import optuna
import os



# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

# Define the PINN
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# Define the network architecture
#layers = [2, 40, 40, 40, 40, 40, 40, 40, 4]  # Input: (x, y), Output: (u, v, p)
layers = [2, 10, 10, 10, 10, 10, 10, 10, 4]  # Input: (x, y), Output: (u, v, p)
#layers = [2, 20, 20, 20, 20, 4]  # Input: (x, y), Output: (u, v, p)
model = PINN(layers).to(device)

def normal_inputs(df): #df is a dataframe
    normal_df = (2 * (df - df.min()) / (df.max() - df.min() )) - 1
    return normal_df
#stage2-1 : Add gausian noise to data for improving and boosting the model
def add_gaussian_noise(tensor, mean=1, std_dev=1.01):
    noise = torch.normal(mean, std_dev, size=tensor.shape)
    return tensor + noise


def navier_stokes_loss(model, x, y):
    x = normal_inputs(x)
    y = normal_inputs(y)
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)

    uvp = model(torch.cat((x, y), dim=1))
    u = uvp[:, 0]
    v = uvp[:, 1]
    p = uvp[:, 2]
    T = uvp[:, 3]

    # Calculate gradients
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True, retain_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True, retain_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]

    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True , retain_graph=True)[0]

    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True, retain_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), create_graph=True, retain_graph=True)[0]

    mu = 0.01
    # Navier-Stokes equations
    f_u = u * u_x + v * u_y + p_x - mu * (u_xx + u_yy)
    f_v = u * v_x + v * v_y + p_y - mu * (v_xx + v_yy)

    # Continuity equation
    continuity = u_x + v_y

    alpha = 0.002
    #Energy Equation
    energy_residual = u * T_x + v * T_y  - alpha * (T_xx + T_yy ) #- alpha_t_diff

    loss_mse = nn.MSELoss()
    #Note our target is zero. It is residual so we use zeros_like
    loss_f = loss_mse(continuity,torch.zeros_like(continuity)) + loss_mse(f_u,torch.zeros_like(f_u)) + loss_mse(f_v,torch.zeros_like(f_v))

    # Loss calculation with balancing factors
    return loss_f

def data_loss(model, file_data):

    df = normal_inputs(pd.read_csv(file_data))
    x_exact = torch.tensor(df[['x']].values, dtype=torch.float32 , requires_grad = True)
    y_exact = torch.tensor(df[['y']].values, dtype=torch.float32 , requires_grad = True)
    exact = torch.tensor(df[['u' ,'v' ,'p' , 'T']].values, dtype=torch.float32 , requires_grad = True)
    uvp_pred = model(torch.cat((x_exact , y_exact) , dim = 1))

    u_pred = uvp_pred[:, 0]
    v_pred = uvp_pred[:, 1]
    p_pred = uvp_pred[:, 2]
    T_pred = uvp_pred[:, 3]

    loss_mse = nn.MSELoss()
    loss_u = loss_mse(u_pred  , exact[: , 0])
    loss_v = loss_mse(v_pred  , exact[: , 1])
    loss_p = loss_mse(p_pred  , exact[: , 2])
    loss_T = loss_mse(T_pred  , exact[: , 3])

    return loss_u + loss_v + loss_p + loss_T

def noisy_data_loss(model, x , y ):

    x_d_noisy = add_gaussian_noise(x.reshape(-1 , 1))
    y_d_noisy = add_gaussian_noise(y.reshape(-1 , 1))

    loss_noisy_PDE = navier_stokes_loss(model , x_d_noisy , y_d_noisy )

    return loss_noisy_PDE

def impose_boundary_conditions(model, x_coords, y_coords , boundary_values):
    # Directly set model predictions to boundary values (hard enforcement)
    with torch.no_grad():
        # Boundary conditions for u, v, T, and p
        model.eval()

        model(torch.cat((x_coords , y_coords) , dim = 1)).copy_(boundary_values)

        model.train()

def plot_results(model,  file_test , epoch):
    model.eval()

    df = normal_inputs(pd.read_csv(file_test))

    x = torch.tensor(df[['x']].values, dtype=torch.float32 , requires_grad = True)
    y = torch.tensor(df[['y']].values, dtype=torch.float32 , requires_grad = True)
    truth = torch.tensor(df[['u' ,'v' ,'p' , 'T']].values, dtype=torch.float32 , requires_grad = True)

    uvpT_pred = model(torch.cat((x , y) , dim = 1))
    u_pred = uvpT_pred[: , 0]
    v_pred = uvpT_pred[: , 1]
    p_pred = uvpT_pred[: , 3]
    T_pred = uvpT_pred[: , 2]

    fig, ax= plt.subplots(nrows=4 , ncols=1  ,dpi = 100 , sharex = True)

    ax[0].set_title("u velocity")
    ax[0].plot(truth[: ,0].detach().numpy() , label = "Exact")
    ax[0].plot(u_pred.detach().numpy() , label= "PINN")
    ax[0].legend(loc="best")


    ax[1].set_title("v velocity")
    ax[1].plot(truth[: ,1].detach().numpy() , label = "Exact")
    ax[1].plot(v_pred.detach().numpy() , label= "PINN")
    ax[1].legend(loc="best")

    ax[2].set_title("p ")
    ax[2].plot(truth[: ,2].detach().numpy() , label = "Exact")
    ax[2].plot(T_pred.detach().numpy() , label= "PINN")
    ax[2].legend(loc="best")

    ax[3].set_title("T")
    ax[3].plot(truth[: ,3].detach().numpy() , label = "Exact")
    ax[3].plot(p_pred.detach().numpy() , label= "PINN")
    ax[3].legend(loc="best")


    fig.suptitle(f'comp PLot : {str(file_test)[:10]}  | EpochNo:{epoch}')
    #fig.savefig("Results/Comparison_plot" + time.strftime("%Y-%m-%d %H%M%S") + ".png")
    plt.show()




def total_loss(model, x_c , y_c ,file_data,
               lambda_PDE,  lambda_data , w_noise ):
    # Physics-informed loss
    loss_f = navier_stokes_loss(model, x_c, y_c) * lambda_PDE
    noisy_pde = noisy_data_loss(model , x_c , y_c) * w_noise
    loss_data = data_loss(model, file_data) * lambda_data        # Data loss

    return loss_f + loss_data + noisy_pde , loss_f , loss_data , noisy_pde



# Optimization using Adam optimizer
def train(model, optimizer, x_c , y_c ,
    file_data,x_coords, y_coords , boundary_values, epochs=10000):

    #Stage4-2 : Hyperparameter tuning with Optuna
    def objective(trial):

        # Hyperparameters for tuning
        lambda_PDE = trial.suggest_float("lambda_PDE", 1, 10)
        lambda_data = trial.suggest_float("lambda_data", 0 ,10)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-7, 1e-2)
        w_noise = trial.suggest_float("w_noise", 0 ,10)


        optimizer = optim.Adam(model.parameters(), lr=learning_rate)


        num_epochs_trial = 500  #best value is 500
        for epoch in range(num_epochs_trial):

            optimizer.zero_grad()

            loss  , loss_PDE , loss_d  , noisy_pde = total_loss(model, x_c , y_c ,file_data,
                           lambda_PDE,  lambda_data , w_noise )

            loss.backward(retain_graph  = True)

            optimizer.step()
            # Return the final loss for this trial
            return loss.item()

    num_trials= 50
    # Run the Optuna hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials = num_trials)  # Adjust n_trials for more thorough search

    os.system('cls')
    # Extract the best lambda values
    best_params = study.best_params
    print("Optimized lambda_pde:", best_params["lambda_PDE"])
    print("Optimized lambda_data:", best_params["lambda_data"])
    print("Optimized Learning Rate:", best_params["learning_rate"])
    print("Optimized weight noisy data:", best_params["w_noise"])


    lambda_PDE = best_params["lambda_PDE"]
    lambda_data = best_params["lambda_data"]
    learning_rate = best_params["learning_rate"]
    w_noise = best_params["w_noise"]


    for epoch in range(epochs):
        optimizer.zero_grad()
        loss  , loss_PDE , loss_d  , noisy_pde = total_loss(model, x_c , y_c ,file_data,
                       lambda_PDE,  lambda_data , w_noise )
        impose_boundary_conditions(model, x_coords, y_coords , boundary_values)

        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.3e} | Loss Data = {loss_d.item():.2e} | Loss PDE = {loss_PDE.item():.2e} | Loss Noisy = {noisy_pde.item():.2e}')
            plot_results(model,  file_data , epoch)



file_data = r"E:\FOAM_PINN\cavHeat\cavity\Data_PINN_Cavity_hetaFlux.csv"
fileBC = r"E:\FOAM_PINN\cavHeat\cavity\BC_PINN_Cavity_hetaFlux.csv"
fileTest = r"E:\FOAM_PINN\cavHeat\cavity\Test_PINN_Cavity_hetaFlux.csv"

#Stage4-0: Collocation points definition

x_min = 0
y_min = 0

x_max = 1
y_max = 0.1

# Generate random collocation points within the domain
np.random.seed(50)
num_collocation_points = 500
collocation_points = np.random.rand(num_collocation_points, 2)
collocation_points[:, 0] = collocation_points[:, 0] * (x_max - x_min) + x_min  # Scale to x bounds
collocation_points[:, 1] = collocation_points[:, 1] * (y_max - y_min) + y_min  # Scale to y bounds

X_c = torch.tensor(collocation_points[:, 0], dtype=torch.float32 ,  requires_grad=True).reshape(-1 , 1)
Y_c = torch.tensor(collocation_points[:, 1], dtype=torch.float32 ,  requires_grad=True).reshape(-1 , 1)

df_boundary = normal_inputs(pd.read_csv(fileBC))
x_coords = torch.tensor(df_boundary[['x']].values, dtype=torch.float32 , requires_grad = True)
y_coords = torch.tensor(df_boundary[['y']].values, dtype=torch.float32 , requires_grad = True)
boundary_values = torch.tensor(df_boundary[['u' ,'v' ,'p' , 'T']].values, dtype=torch.float32 , requires_grad = True)

# Define the model and optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
#optim.Adamax(model.parameters(), lr=1e-2)


# Training parameters
epochs = 15000
mu = 0.01  # Dynamic viscosity
alpha = 0.002
lambda_momentum = 0.3
lambda_continuity = 0.2
lambda_data = 1
w_noise = 1


# Train the model
train(model, optimizer, X_c , Y_c ,
    file_data,x_coords, y_coords , boundary_values, epochs=10000)

plot_results(model,  fileTest , epoch = epochs)

# Plotting predicted vs exact values
uvp_pred = model(torch.cat((x_data, y_data), dim=1)).detach().cpu().numpy()
u_pred = uvp_pred[:, 0]
v_pred = uvp_pred[:, 1]
p_pred = uvp_pred[:, 2] if p_exact is not None else None

u_exact = u_exact.cpu().numpy()
v_exact = v_exact.cpu().numpy()
p_exact = p_exact.cpu().numpy() if p_exact is not None else None

plt.figure()
plt.plot(u_exact, label = 'u Exact')
plt.plot(u_pred, label='u PINN')
plt.legend()

plt.figure()
plt.plot(v_exact, label = 'v Exact')
plt.plot(v_pred, label='v PINN')
plt.legend()


plt.figure()
plt.plot(p_exact, label='p Exacy')
plt.plot(p_pred, label='p PINN')
plt.xlabel('Exact p')
plt.ylabel('Predicted p')
plt.legend()

plt.tight_layout()
plt.show()
