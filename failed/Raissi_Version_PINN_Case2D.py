import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import os
import time
from scipy.interpolate import griddata
import scipy
import tqdm


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Swish activation function
class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x = x * torch.sigmoid(x)  # inplace modification removed for safer execution
            return x
        else:
            return x * torch.sigmoid(x)

# Define the PINN model
class PINN_psi(nn.Module):
    def __init__(self, layers):
        super(PINN_psi, self).__init__()

        # Initialize neural network layers
        self.layers = nn.ModuleList()


        # Dynamically create layers based on input list 'layers'
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        # Apply Xavier initialization to all weights
        self.apply(self.xavier_init)

    # Xavier initialization function
    def xavier_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    # Forward pass using Swish activation
    def forward(self, x):
        for layer in self.layers[:-1]:  # Apply Swish activation for all layers except the last
            x = Swish()(layer(x))

        x = self.layers[-1](x)  # Output layer with no activation
        return x

    def xavier_init(self, layer):
        if isinstance(layer, nn.Linear):
            # Initialize weights with Xavier and set biases to zero
            nn.init.xavier_uniform_(layer.weight)  # or use xavier_normal_
            nn.init.constant_(layer.bias, 0)


class PINN_p(nn.Module):
    def __init__(self, layers):
        super(PINN_p, self).__init__()

        # Initialize neural network layers
        self.layers = nn.ModuleList()


        # Dynamically create layers based on input list 'layers'
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        # Apply Xavier initialization to all weights
        self.apply(self.xavier_init)

    # Xavier initialization function
    def xavier_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    # Forward pass using Swish activation
    def forward(self, x):
        for layer in self.layers[:-1]:  # Apply Swish activation for all layers except the last
            x = Swish()(layer(x))

        x = self.layers[-1](x)  # Output layer with no activation
        return x

    def xavier_init(self, layer):
        if isinstance(layer, nn.Linear):
            # Initialize weights with Xavier and set biases to zero
            nn.init.xavier_uniform_(layer.weight)  # or use xavier_normal_
            nn.init.constant_(layer.bias, 0)

class PINN_T(nn.Module):
    def __init__(self, layers):
        super(PINN_T, self).__init__()

        # Initialize neural network layers
        self.layers = nn.ModuleList()


        # Dynamically create layers based on input list 'layers'
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        # Apply Xavier initialization to all weights
        self.apply(self.xavier_init)

    # Xavier initialization function
    def xavier_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    # Forward pass using Swish activation
    def forward(self, x):
        for layer in self.layers[:-1]:  # Apply Swish activation for all layers except the last
            x = Swish()(layer(x))

        x = self.layers[-1](x)  # Output layer with no activation
        return x


# use the modules apply function to recursively apply the initialization
#net1 = Net1().to(device)
# Example usage
NUM_NEURONS = int(20)
NUM_LAYER = 8
dim_input = int(2)
dim_output = int(1)
layers = np.zeros(NUM_LAYER)
layers = [dim_input ]

for i in range(1 , NUM_LAYER+1):
    layers.append(NUM_NEURONS)
    if i==NUM_LAYER:
        layers.append(dim_output)


#print(layers)
model_psi = PINN_psi(layers).to(device)
model_p = PINN_p(layers).to(device)
model_T = PINN_T(layers).to(device)



##############################################################################
def add_gaussian_noise(tensor, mean=1, std_dev=1.01):
    noise = torch.normal(mean, std_dev, size=tensor.shape)
    return tensor + noise

def normal_inputs(df): #df is a dataframe
    normal_df = (2 * (df - df.min()) / (df.max() - df.min() )) - 1
    return normal_df

def pde_residuals(model_psi , model_p  , model_T, x , y ):

    psi = model_psi(torch.cat((x,y) , dim = 1))
    p = model_p(torch.cat((x,y) , dim = 1))
    T = model_T(torch.cat((x,y) , dim = 1))

    u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True )[0]
    v = -torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True )[0]

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True )[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True )[0]
    u_xx = torch.autograd.grad(u_x , x, grad_outputs=torch.ones_like(u_x), create_graph=True )[0]
    u_yy = torch.autograd.grad(u_y , y, grad_outputs=torch.ones_like(u_x), create_graph=True )[0]

    v_x = torch.autograd.grad(v , x , grad_outputs=torch.ones_like(u), create_graph=True )[0]
    v_y = torch.autograd.grad(v , y , grad_outputs=torch.ones_like(u), create_graph=True )[0]
    v_xx = torch.autograd.grad(v_x , x, grad_outputs=torch.ones_like(u_x), create_graph=True )[0]
    v_yy = torch.autograd.grad(v_y , y, grad_outputs=torch.ones_like(u_x), create_graph=True )[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True )[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True )[0]

    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True )[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True )[0]

    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True )[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), create_graph=True )[0]

    alpha = 0.002
    mu = 0.01
    continuity_residual = u_x + v_y
    momentum_u_residual =  lambda_1 * (u*u_x + v*u_y) + p_x - mu * lambda_2 * (u_xx + u_yy)
    momentum_v_residual =  lambda_1 * (u*v_x + v*v_y) + p_y - mu * lambda_2 * (v_xx + v_yy)
    energy_residual = lambda_3 *(u * T_x + v * T_y)  - alpha * lambda_4 * (T_xx + T_yy ) #- alpha_t_diff

    loss_mse = nn.MSELoss()
    #Note our target is zero. It is residual so we use zeros_like
    loss_pde = loss_mse(continuity_residual,torch.zeros_like(continuity_residual)) + loss_mse(momentum_u_residual,torch.zeros_like(momentum_u_residual)) + loss_mse(momentum_v_residual,torch.zeros_like(momentum_v_residual)) + loss_mse(energy_residual,torch.zeros_like(energy_residual))
    return loss_pde


def data_loader(file_name):
    df =  normal_inputs(pd.read_csv(file_name))
    x = torch.tensor(df[['x']].values, dtype=torch.float32 , requires_grad = True).reshape(-1 , 1)
    y = torch.tensor(df[['y']].values, dtype=torch.float32 , requires_grad = True).reshape(-1 , 1)
    truth = torch.tensor(df[['u' ,'v' ,'T' , 'p']].values, dtype=torch.float32 , requires_grad = True).reshape(-1 , 4)

    return x , y , truth


def data_loss(model_psi , model_p , model_T , x , y , truth):
    loss_mse = nn.MSELoss()

    psi = model_psi(torch.cat((x,y) , dim = 1))
    p_pred = model_p(torch.cat((x,y) , dim = 1))
    T_pred = model_T(torch.cat((x ,y) , dim = 1))

    u_pred = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True )[0]

    u_truth = truth[: , 0].reshape(-1 , 1)
    u_loss = loss_mse(u_pred, u_truth)

    p_truth = truth[: , 3].reshape(-1 , 1)
    p_loss = loss_mse(p_pred, p_truth)

    T_truth = truth[: , 2].reshape(-1 , 1)
    T_loss = loss_mse(T_pred, T_truth)

    loss = u_loss + p_loss + T_loss


    return loss , u_loss , p_loss , T_loss

def impose_boundary_conditions(model_psi , model_p , model_T , fileBC):
    # Directly set model predictions to boundary values (hard enforcement)
    with torch.no_grad():

        x_coords , y_coords , boundary_values = data_loader(fileBC)

        # Boundary conditions for u, v, T, and p
        model_psi.eval()
        model_p.eval()
        model_T.eval()

        model_psi(torch.cat((x_coords , y_coords) , dim = 1)).copy_(boundary_values[: , 0].reshape(-1 , 1))
        model_p(torch.cat((x_coords , y_coords) , dim = 1)).copy_(boundary_values[: , 3].reshape(-1 , 1))
        model_T(torch.cat((x_coords , y_coords) , dim = 1)).copy_(boundary_values[: , 2].reshape(-1 , 1))

        model_psi.train()
        model_p.train()
        model_T.train()

def collocation_points(x_min , y_min , x_max , y_max , cube_x_min, cube_x_max ,
                        cube_y_min, cube_y_max , num_collocation_points ):
    #Stage4-0: Collocation points definition
    # Generate random collocation points within the domain
    np.random.seed(50)

    collocation_points = np.random.rand(num_collocation_points, 2)
    collocation_points[:, 0] = collocation_points[:, 0] * (x_max - x_min) + x_min  # Scale to x bounds
    collocation_points[:, 1] = collocation_points[:, 1] * (y_max - y_min) + y_min  # Scale to y bounds

    # Filter out points that fall within the cube's region
    filtered_points = collocation_points[
        ~(
        (collocation_points[:, 0] >= cube_x_min) &
        (collocation_points[:, 0] <= cube_x_max) &
        (collocation_points[:, 1] >= cube_y_min) &
        (collocation_points[:, 1] <= cube_y_max)
        )]


    collocation_points_tensor = torch.tensor(filtered_points, dtype=torch.float32 ,  requires_grad=True)
    X_c = collocation_points_tensor[: , 0].reshape(-1 , 1)
    Y_c = collocation_points_tensor[: , 1].reshape(-1 , 1)

    return X_c, Y_c

def noisy_data_loss(model_psi , model_p , model_T , x , y ):

    x_noisy = add_gaussian_noise(x.reshape(-1 , 1))
    y_noisy = add_gaussian_noise(y.reshape(-1 , 1))

    noisy_resid_loss = pde_residuals(model_psi , model_p  , model_T, x_noisy , y_noisy )



    return noisy_resid_loss

def total_loss(model_psi , model_p , model_T , x_c , y_c ,
               x_int , y_int , truth_int , x_noisy , y_noisy ):
    pde_loss = pde_residuals(model_psi , model_p  , model_T, x_c , y_c )
    loss_data , _ , _ , _ = lambda_data * data_loss(model_psi , model_p , model_T , x_int , y_int , truth_int)
    noisy_loss = lambda_noisy * noisy_data_loss(model_psi , model_p , model_T, x_noisy , y_noisy )

    loss = pde_loss + loss_data + noisy_loss
    return loss , pde_loss , loss_data , noisy_loss

def plot_solution(x_star,y_star , u_star , title):

    lb = [min(x_star) , min(y_star)]
    ub = [max(x_star) , max(y_star)]

    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    points = np.column_stack((x_star.flatten(), y_star.flatten()))


    U_star = griddata(points, u_star.flatten(), (X, Y), method='cubic')
    
    
    plt.figure(dpi = 100)
    plt.pcolor(X,Y,U_star,  vmin=-1, vmax=1 , cmap = 'jet')
    plt.title(title)
    plt.colorbar()
    plt.grid()
    plt.show()

def train(model_psi , model_p , model_T , fileData , fileBC , nIter):

    x_min = 0
    y_min = 0

    x_max = 10
    y_max = 5
    num_collocation_points = 500
    # Define cube boundaries within the domain (example values)
    cube_x_min, cube_x_max = 4.5, 5.5  # x bounds of cube
    cube_y_min, cube_y_max = 0, 1  # y bounds of cube
    X_c , Y_c = collocation_points(x_min , y_min , x_max , y_max , cube_x_min, cube_x_max ,
                        cube_y_min, cube_y_max , num_collocation_points )
    x_inter , y_inter , truth_inter = data_loader(fileData)
    x_boundary , y_boundary , truth_boundary = data_loader(fileBC)
    x_noisy, y_noisy = add_gaussian_noise(x_inter), add_gaussian_noise(y_inter)

    optimizer_psi = torch.optim.Adam(model_psi.parameters(), lr=1e-3)
    optimizer_p = torch.optim.Adam(model_p.parameters(), lr=1e-3)
    optimizer_T = torch.optim.Adam(model_T.parameters(), lr=1e-3)


    loss_hist = []
    for epoch in range(nIter):

        model_psi.train()
        model_p.train()
        model_T.train()

        optimizer_psi.zero_grad()
        optimizer_p.zero_grad()
        optimizer_T.zero_grad()

        loss , pdeLOSS , dataLOSS , noisyLOSS  = total_loss(model_psi , model_p , model_T , X_c , Y_c ,
               x_inter , y_inter , truth_inter , x_noisy , y_noisy )
        impose_boundary_conditions(model_psi , model_p , model_T , fileBC)



        loss.backward()

        optimizer_psi.step()
        optimizer_p.step()
        optimizer_T.step()

        loss_hist.append(loss.item())

        if epoch % 250 == 0:
            print(f'Epoch Adam {epoch}/{nIter} [{100 * epoch/nIter :.2f}%] || Train Loss: {loss.item():.3e}  ')
            print(f'PDE Loss: {pdeLOSS.item():.3e}, Data Loss: {dataLOSS.item():.2e}, Noisy Loss: {noisyLOSS.item():.2e}')
    
    xepoch = np.linspace(0 , nIter , len(loss_hist))
    snap = int(0.5 * len(loss_hist))
    plt.plot(xepoch[snap:] , loss_hist[snap:] , label = "loss")
    plt.title("Loss history")
    plt.legend()
    plt.show()
    


lambda_1 = 1
lambda_2 = 1
lambda_3 = 1
lambda_4 = 1
lambda_data = 1
lambda_noisy = 1

#file_data = r"2D_newData.csv"
file_data = r"2D_contur_Box.csv"
fileBC = r"BC_data_2D_Lamin.csv"
fileTest = r"2D_newTest.csv"

train(model_psi , model_p , model_T , file_data , fileBC , 20000)

x , y , truth = data_loader(file_data)

psi_star = model_psi(torch.cat((x , y) , dim = 1))
u_star = torch.autograd.grad(psi_star, y, grad_outputs=torch.ones_like(psi_star), create_graph=True )[0]
T_star = model_T(torch.cat((x , y) , dim = 1))
p_star = model_p(torch.cat((x , y) , dim = 1))
x.detach().numpy()
y.detach().numpy()
u_star.detach().numpy()
T_star.detach().numpy()

plot_solution(x.detach().numpy() , y.detach().numpy() , p_star.detach().numpy() , "PINN")
plot_solution(x.detach().numpy() , y.detach().numpy() , truth[: , 3].detach().numpy() , "Truth")


plt.figure()
plt.plot(p_star.detach().numpy() , label = "PINN")
plt.plot(truth[: , 3].detach().numpy() , label = "Truth")
plt.legend()
plt.ylim(-1 , 1)

plt.figure()
plt.plot(u_star.detach().numpy() , label = "PINN")
plt.plot(truth[: , 0].detach().numpy() , label = "Truth")
plt.legend()
plt.ylim(-1 , 1)
