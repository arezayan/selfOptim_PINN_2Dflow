# [2D PINN Solver: Physics-Informed Neural Networks for 2D Problems](https://github.com/arezayan/selfOptim_PINN_2Dflow/)

> **Important:** *This repository provides a Python-based implementation of a 2D Physics-Informed Neural Network (PINN) designed to solve Partial Differential Equations (PDEs) in two dimensions. The code is structured for flexibility, enabling users to define custom governing equations, boundary conditions, and datasets.*

> # The core features include:

* Support for Continuity, Momentum, and Energy equations.
* Dynamic learning rate scheduling for optimized training.
* Implementation of PDE loss, data loss, and boundary condition loss with proper weight balancing.
* Use of just Adam optimizaer method .
* Compatibility with CSV-based datasets for boundary and interior data.
> # Features:
* Governing Equations:
* Solves custom PDEs such as the Navier-Stokes and Energy equations.
> # Loss Functions:
* Includes separate loss components for:
>> * PDE residuals
>> * Interior points
>> * explicity imposed Boundary conditions


> # Data Handling:
>>* Supports loading boundary and interior data from CSV files.

## Installation
  
  
> * 1- Clone the repository:

```html
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```
> * 2- Install dependencies:
```html
pip install -r requirements.txt
```


