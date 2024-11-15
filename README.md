# Solving High Dimensional PDEs with BSDEs Using PyTorch

- This repository contains Python scripts that implement solutions to backward stochastic differential equations (BSDEs) for several high-dimensional partial differential equations using PyTorch.

- We draw an analogy between BSDEs and reinforcement learning, where the gradient of the solution serves as the policy function, and the loss function is defined by the discrepancy between the specified terminal condition and the solution of the BSDE.

## Quick start

#### Install with environment

```python
conda create -n pdes python
# https://pytorch.org/
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install scipy
conda install matplotlib
```

#### python code

```python
from equation import HJBLQ
from default_parameters import HJBConfig
from train import BSDESolver
HJB = HJBLQ(HJBConfig)
model = BSDESolver(HJBConfig, HJB)
model.solve()
```

#### Folder Structure

`lib.py` : Some custom functions, i.e. Early Stopping and Loading PyTorch models.

`main.py : ` The starting point to run the program.

`train.py :` Deep Learning models of Deep_BSDE algorithm.

`equation.py :` This file contains the definition of the partial differential equation and any necessary boundary conditions.

`saved_models :` The trained models to be saved for next execution.

`default_parameters.py :` Configurations for equation construction and training process.

#### Example 

**The Allen Cahn equation** is a mathematical model that describes the evolution of an interface during a phase transition. This equation is especially suitable for simulating the interfacial dynamics in nonlinear systems, such as phase transformation in alloys and morphological changes of biofilms. It is a typical nonlinear partial differential equation, which can describe the transition from a uniform state to a non-uniform state.

<img src="https://img2.imgtp.com/2024/04/05/OMqP4PA7.png" alt="The prediction of Allen Cahn equation">


## Theoretic Background

Based on the backward stochastic differential equation (BSDE) representation of PDEs, we use deep neural network to estimate the solutions and gradients of the equations at the same time. By using the nonlinear Feynman-KAC formula, the solution of the high-dimensional PDEs can be expressed as the solution of the corresponding BSDE equations. Then the numerical problem is expressed as a stochastic control problem, and the gradient operator of the solution function is regarded as a policy function, and this policy function is approximated by a deep neural network, and then the numerical solution of the high-dimensional PDEs is obtained.[2]

So, in the general case, we are interested in the semilinear parabolic PDEs.The following equations of are implemented in this repository:

- Allen-Cahn Equationf
- Hamilton-Jacobi-Bellman Equation
- Nonlinear Black-Scholes Equation with Default Risk
- Pricing European Financial Derivatives with Different Interest Rates for Borrowing and Lending
- Multidimensional Burgers-Type PDEs with Explicit Solutions
- Reaction Diffusion Time Dependent Example PDE with Oscillating Explicit Solutions



## Reference

- [1] Weinan E, Han J, Jentzen A. Algorithms for solving high dimensional PDEs: from nonlinear Monte Carlo to machine learning[J]. Nonlinearity, 2021, 35(1): 278.

- [2] Han J, Jentzen A, E W. Solving high-dimensional partial differential equations using deep learning[J]. Proceedings of the National Academy of Sciences, 2018, 115(34): 8505-8510.

- [3] Han J, Jentzen A. Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations[J]. Communications in mathematics and statistics, 2017, 5(4): 349-380.
