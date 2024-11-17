
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os
import matplotlib




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
NUM_NEURONS = int(40)
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

save_to_path = 'E:/FOAM_PINN/cavHeat/twoD_lamin_over_box/raiss_results/'
model_PSI_name = 'raiss_results/saved_model_psi.pth'
model_p_name =  'raiss_results/saved_model_p.pth'
model_T_name =  'raiss_results/saved_model_T.pth'


file_data = r"2D_newData.csv"
#file_data = r"data_2D_Lamin.csv"
fileBC = r"BC_data_2D_Lamin.csv"
#fileTest = r"2D_newTest.csv"
#fileTest = r"paraFoam_test.csv"
fileTest = r"randomPoints.csv"



def simple_data_loader(file_name):
    df =  normal_inputs(pd.read_csv(file_name))
    x = torch.tensor(df[['x']].values, dtype=torch.float32 , requires_grad = True).reshape(-1 , 1)
    y = torch.tensor(df[['y']].values, dtype=torch.float32 , requires_grad = True).reshape(-1 , 1)
    truth = torch.tensor(df[['u' ,'v' ,'T' , 'p']].values, dtype=torch.float32 , requires_grad = True).reshape(-1 , 4)

    return x , y , truth

def load_model(file_pth , model , layers):
    init_model = model(layers)
    state_dict = torch.load(file_pth , map_location = device)
    #state_dict = torch.load(file_pth
    init_model.load_state_dict(state_dict)
    return init_model #LOADED model



def normal_inputs(df): #df is a dataframe
    normal_df = (2 * (df - df.min()) / (df.max() - df.min() )) - 1
    return normal_df

def plot_solution(x_star,y_star , u_star , title):

    lb = [min(x_star) , min(y_star)]
    ub = [max(x_star) , max(y_star)]

    nn = 100
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
    plt.savefig(save_to_path + title +  ".png")
    #plt.show()





##############################################################################
model_psi = load_model(model_PSI_name , PINN_psi , layers)
model_p = load_model(model_p_name , PINN_p , layers)
model_T = load_model(model_T_name , PINN_T , layers)


x , y , truth = simple_data_loader(fileTest)

psi_star = model_psi(torch.cat((x , y) , dim = 1))
u_star = torch.autograd.grad(psi_star, y, grad_outputs=torch.ones_like(psi_star), create_graph=True )[0]
Tt_star = model_T(torch.cat((x , y) , dim = 1))
p_star = model_p(torch.cat((x , y) , dim = 1))
x.detach().numpy()
y.detach().numpy()
u_star.detach().numpy()
Tt_star.detach().numpy()



x = x.detach().numpy()
y = y.detach().numpy()
truth.detach().numpy()

lb = [min(x) , min(y)]
ub = [max(x) , max(y)]

nn = 220

X, Y = np.meshgrid(x,y)
points = np.column_stack((x.flatten(), y.flatten()))

U_star = griddata(points, u_star.flatten().detach().numpy(), (X, Y), method='cubic')
P_star = griddata(points, p_star.flatten().detach().numpy(), (X, Y), method='cubic')
T_star = griddata(points, Tt_star.flatten().detach().numpy(), (X, Y), method='cubic')
u_truth = griddata(points, truth[: , 0].flatten().detach().numpy(), (X, Y), method='cubic')
p_truth = griddata(points, truth[: , 3].flatten().detach().numpy(), (X, Y), method='cubic')
T_truth = griddata(points, truth[: , 2].flatten().detach().numpy(), (X, Y), method='cubic')

fig, axes = plt.subplots(3, 3 , dpi = 150 , figsize = (18 , 9))
parameters = ['u velocity', 'Temperature (T)' , 'pressure (p)']
matplotlib.rcParams.update({'font.size': 10})


pcol0 = axes[0 , 0].pcolor(X,Y,U_star,  vmin=-1, vmax=1 , cmap = 'jet')
axes[0 , 0].set_title(f'PINN {parameters[0]}' )
axes[0 , 0].set_xlabel('x')
axes[0 , 0].set_ylabel('y')
fig.colorbar(pcol0)

pcol1 = axes[0 , 1].pcolor(X, Y, u_truth, shading='auto', cmap='jet')
axes[0 , 1].set_title(f'Truth {parameters[0]}')
axes[0 , 1].set_xlabel('x')
axes[0 , 1].set_ylabel('y')
fig.colorbar(pcol1)

axes[0 , 2].plot(u_star.detach().numpy() , marker = "+" ,  label="PINN")
axes[0 , 2].plot(truth[: , 0].detach().numpy() , label = "Truth")
axes[0 , 2].set_title(f'Line Comparison {parameters[0]}')
axes[0 , 2].set_xlabel('x')
axes[0 , 2].set_ylabel(parameters[0])
axes[0 , 2].legend()



pcolT0 = axes[1 , 0].pcolor(X,Y,P_star,  vmin=-1, vmax=1 , cmap = 'jet')
axes[1 , 0].set_title(f'PINN {parameters[2]}')
axes[1 , 0].set_xlabel('x')
axes[1 , 0].set_ylabel('y')
fig.colorbar(pcolT0)

pcolT1 =  axes[1 , 1].pcolor(X, Y, p_truth, shading='auto', cmap='jet')
axes[1 , 1].set_title(f'Truth {parameters[2]}')
axes[1 , 1].set_xlabel('x')
axes[1 , 1].set_ylabel('y')
fig.colorbar(pcolT1)

axes[1 , 2].plot(p_star.detach().numpy() , marker = "+" ,  label="PINN")
axes[1 , 2].plot(truth[: , 3].detach().numpy() , label = "Truth")
axes[1 , 2].set_title(f'Line Comparison {parameters[2]}')
axes[1 , 2].set_xlabel('x')
axes[1 , 2].set_ylabel(parameters[2])
axes[1 , 2].legend()

pcolp0 = axes[2 , 0].pcolor(X,Y,T_star,  vmin=-1, vmax=1 , cmap = 'jet')
axes[2 , 0].set_title(f'PINN {parameters[1]}')
axes[2 , 0].set_xlabel('x')
axes[2 , 0].set_ylabel('y')
fig.colorbar(pcolp0 )

pcolp1 = axes[2 , 1].pcolor(X, Y, T_truth, shading='auto', cmap='jet')
axes[2 , 1].set_title(f'Truth {parameters[1]}')
axes[2 , 1].set_xlabel('x')
axes[2 , 1].set_ylabel(parameters[1])
fig.colorbar(pcolp1)

axes[2 , 2].plot(Tt_star.detach().numpy() , marker = "+" ,  label="PINN")
axes[2 , 2].plot(truth[: , 2].detach().numpy() , label = "Truth")
axes[2 , 2].set_title(f'Line Comparison {parameters[1]}')
axes[2 , 2].set_xlabel('x')
axes[2 , 2].set_ylabel(parameters[1])
axes[2 , 2].legend()

fig.savefig(save_to_path + "3by3_PINN_2D.png")

plot_solution(x , y , u_star.detach().numpy() , "PINN")
plot_solution(x , y , truth[: , 0].detach().numpy() , "Truth")
