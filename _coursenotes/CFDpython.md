---
title: "Computational Fluid Dynamics Python (CFD) - Python"
categories:
toc: true
layout: single
permalink: /coursenotes/CFDpython/
author_profile: true
read_time: true
---

Class notes for the CFD-Python course taught by Prof. Lorena Barba, Boston University. The course outlines a 12 step program, each of increasing difficulty, towards building mastery solving the Navier-Stokes equations through finite difference techniques.  
$$
\begin{align}
\dfrac{\partial \vec{u}}{\partial t} +(\vec{u}\cdot\vec{\nabla})\vec{u} = -\dfrac{\vec{\nabla}P}{\rho}+\nu\vec{\nabla}^2\vec{u} + \sum\vec{F}_{\textrm{ext}} \label{eqn:NS}
\end{align}
$$



## Preliminaries

The Naiver-Stokes (N-S) equation represents the conservation of momentum of purely viscous, i.e. Newtonian fluids, written in differential form.[^1] For most realistic fluid transport problems, it is difficult to find analytical solutions to the N-S equation. In this class, we learn how to implement finite-difference based numerical solutions to the N-S equations using `python`, making heavy use of the `numpy` and `matplotlib` libraries.

The finite difference technique involves approximating the differential operator by making use Taylor series expansions. Given a differentiable function $f(x)$, we can use the Taylor series to approximate the function around the point $x = x_0$ as follows:

$$
\begin{align}
f(x_0 + \Delta x) = f(x_0) + \Delta x\left.\dfrac{\partial f(x)}{\partial x}\right|_{x = x_0} + O(\Delta x^2) \label{eqn:Taylor}
\end{align}
$$

If $\Delta x$ is small enough, the higher order terms can be made negligibly small and hence we can rearrange terms in Equation \eqref{eqn:Taylor} to approximate the derivative as

$$
\begin{align}
\dfrac{\partial f(x)}{\partial x} = \dfrac{f(x + \Delta x) - f(x)}{\Delta x} + O(\Delta x) \label{eqn:TaylorDiff}
\end{align}
$$

This is a fundamental result in the finite difference numerical scheme from which many other approxiations to higher order derivatives can be derived.

## Step 1: 1-D Linear Convection

We study how to implement the approximation shown in Equation \eqref{eqn:TaylorDiff} using the simplest example of 1-D linear convection, i.e., the 1-D wave equation, which is given by

$$
\begin{align}
\dfrac{\partial u}{\partial t} + c\dfrac{\partial u}{\partial x} = 0 \label{eqn:wave}
\end{align}
$$

Here, $u(x, t)$ represents the velocity at position $x$ and time $t$, and $c$ is the wave propagation speed. We first discretize $t$ and $x$ into $t_0, t_1, \ldots, t_N$ and $x_0, x_1, \ldots, x_M$ points respectively. Let $t_{i+1} - t_{i} = \Delta t$ and $x_{j+1} - x_{j} = \Delta x$ for all $i \in {0, \ldots, N}$ and $i \in {0, \ldots, M}$. We use the notation $u_i^n$ to represent $u(x_i, t_n)$. Then, using Equation \eqref{eqn:TaylorDiff} we have

$$
\begin{align}
\dfrac{\partial u}{\partial t} & \approx \dfrac{u_i^{n+1}-u_{i}^n}{\Delta t} \label{eqn:timeapprox} \\
\dfrac{\partial u}{\partial x} & \approx \dfrac{u_i^n-u_{i-1}^n}{\Delta x} \label{eqn:xapprox}
\end{align}
$$

and therefore the discretized wave equation becomes

$$
\begin{align}
\dfrac{u_i^{n+1}-u_{i}^n}{\Delta t} + c\dfrac{u_i^n-u_{i-1}^n}{\Delta x} = 0 \label{eqn:wave-equation-discrete}
\end{align}
$$

Note that we use a _forward difference_ approximation for time and a _backward difference_ approximation for space. We have therefore converted the wave equation from a differential equation to a system of algebraic equations which can be solved with standard matrix methods. Given appropriate initial and boundary conditions,  the only unknown at each time point is $u^{n+1}_i$ and we can use a time-marching Euler scheme to solve for $u$ at every discrete point in time and space. 

To see this, consider Equation \eqref{eqn:wave-equation-discrete} for the case $i=1$ and $n=0$ (the case $i=0$ and $n=0$ are known from initial and boundary conditions):

$$
\begin{align}
\dfrac{u_1^{1}-u_{1}^0}{\Delta t} + c\dfrac{u_1^0-u_{0}^0}{\Delta x} = 0 
\end{align}
$$

which can rewritten as

$$
\begin{align}
u_1^1 = u_1^0 -c\dfrac{\Delta t}{\Delta x}(u_1^0-u_{0}^0) 
\end{align}
$$

Because all the terms in the RHS are known, we can find $u_1^1$, and extending this scheme, we find $u_2^1, u_3^1, \ldots, u_M^1$. Having found the velocity all spatial grid points, we next march forward in time to $n=2$ and repeat the process, until $u$ is found for all time and space points in the grid. 

**Exercise**

We implement the above scheme by solving Equation \eqref{eqn:wave-equation-discrete} using the initial conditions

$$
\begin{align}
u(x, 0) = 
\begin{cases}
2, 0.5 \leq x \leq 1, \\
1, \textrm{otherwise}
\end{cases} \label{eqn:ICs}
\end{align}
$$

and the boundary condition $u(0, t) = u(2, t) = 1$.

```python
import numpy as np
from matplotlib import pyplot as plt

# Initializing parameters
nx = 41  
dx = 2 / (nx-1)
nt = 25
dt = .025
c = 1   

# Setting up the intial and boundary conditions
u = numpy.ones(nx)      #numpy function ones()
u[int(.5 / dx):int(1 / dx + 1)] = 2  #setting u = 2 between 0.5 and 
                                     # 1 as per our I.C.s

# Calculating solution
un = u.copy()

for n in range(nt):  #loop for values of n from 0 to nt
    u[1:] = un[1:] - c * dt/ dx * (un[1:] - un[:-1])
```

Note that we have made use to `numpy`'s broadcasting property to calculate $u^n_i - u^n_{i-1}$. We now plot our solution:

```python
plt.plot(np.linspace(0, 2, nx), u);
```

![test image size](/assets/images/coursenotes/CFDpython/step1.png){: width="75%" .align-center}

Here we see something strange! Our intial conditions suggest that we had a square pulse, and given the nature of the wave equation (with no diffusion or convection for now), we would expect this square to simply translate along x with time, but we see a rounded square here! This has happened because of numerical diffusion: our values of $\Delta x$ were too large. Indeed, we can make $\Delta x$ smaller and check that the intial square pulse translates as a square. 

We made some arbitrary choices around the values of $\Delta x$ and $\Delta t$. Playing with the code, you will see that for some values of the $\Delta x$ and $\Delta t$, the solution seems unstable and in reality, we need to be careful about the how we go about choosing $\Delta x$ and $\Delta t$. We require that the distance traveled $u\Delta t$ travelled by the wave in the time interval $\Delta t$ be less than the chosen value of $\Delta x$. There is detailed theory behind this, capture by the Courant-Friendrichs-Lewy (CLF) condition.[^2] Formally, we require

$$
\begin{align}
\dfrac{u\Delta t}{\Delta x} \leq \sigma_\textrm{max}
\end{align}
$$

where $\sigma_\textrm{max} \leq 1$ is the Courant number whose value changes with the exact numerical method being used. For the explicit Euler scheme described above, $\sigma_\textrm{max} \approx 1$. We will see more about this later while looking at solving equations in higher dimensions.

## Step 2: Nonlinear 1-D convection

The equation for nonlinear 1-D convection is

$$
\begin{align}
\dfrac{\partial u}{\partial t} + u\dfrac{\partial u}{\partial x} = 0
\end{align}
$$

Using the ideas above, it is simple to discretize this equation, and solving for the unkown term $u_i^{n+1}$, we have

$$
\begin{align}
u_i^{n+1} = u_i^{n}  -\dfrac{\Delta t}{\Delta x} u_i^{n}(u_i^{n} - u_{i-1}^{n})
\end{align}
$$

We use the same initial conditions as in Equation \eqref{eqn:ICs}. This is fairly easy to implement and very similar to the simple 1-D linear convection case above, with the replace $c \rightarrow u$. We therefore omit the exercise here. 

## Step3: 1-D Diffusion

For constant diffusivity $\nu$, the 1-D diffusion equation is

$$
\begin{align}
\dfrac{\partial u}{\partial t} = \nu\dfrac{\partial^2 u}{\partial t^2} \label{eqn:diffusion1d}
\end{align}
$$

We now need to discretize the second order derivative. Again, we start with the Taylor series expansion for $u = u_{i+1}$ and $u = u_{i-1}$ around $u = u_i$:

$$
\begin{align}
u_{i+1} = u_i + \Delta x \left.\dfrac{\partial u}{\partial x}\right|_{i} + 
            \dfrac{\Delta x^2}{2!} \left.\dfrac{\partial^2 u}{\partial x^2}\right|_{i} + 
            \dfrac{\Delta x^3}{3!} \left.\dfrac{\partial^3 u}{\partial x^3}\right|_{i} + 
            O(\Delta x^4) \\
u_{i-1} = u_i - \Delta x \left.\dfrac{\partial u}{\partial x}\right|_{i} + 
            \dfrac{\Delta x^2}{2!} \left.\dfrac{\partial^2 u}{\partial x^2}\right|_{i} -
            \dfrac{\Delta x^3}{3!} \left.\dfrac{\partial^3 u}{\partial x^3}\right|_{i} + 
            O(\Delta x^4) 
\end{align}
$$

Adding these two equations, we find that

$$
\begin{align}
u_{i+1} + u_{i-1} & = 2\left( u_i + \dfrac{\Delta x^2}{2!} \left.\dfrac{\partial^2 u}{\partial x^2}\right|_{i} \right) + 
                    O(\Delta x^4) \\
\Rightarrow \left.\dfrac{\partial^2 u}{\partial x^2}\right|_{i} & = 
            \dfrac{u_{i+1} -2u_i + u_{i-1}}{\Delta x^2} + O(\Delta x^2) \label{eqn:second-order-discrete}
\end{align}
$$

The terms on the LHS tells us that we have performed a *central difference* (CD), as opposed to the forward and backward differemce, FD anf BD, we have seen so far. The CD ignores terms of $O(\Delta x^2)$, while FD and BD ignore terms of order $O(\Delta x)$ and hence has higher accuracy. Discretizing the equations by using Equations \eqref{eqn:timeapprox} and \eqref{eqn:second-order-discrete}  in \eqref{eqn:diffusion1d} and rearranging terms, we have

$$
\begin{align}
u_i^{n+1} = u_i^n + \dfrac{\nu \Delta t}{\Delta x^2}(u_{i+1}^n - 2 u_i^n + u_{i-1}^n) \label{eqn:1ddiffusion-discrete}
\end{align}
$$

along with the same initial conditions as in Equation \eqref{eqn:ICs}. Because this equation is in 2_d, the CLF condition is modified, and now becomes:

$$
\begin{align}
\dfrac{\nu \Delta t}{\Delta x} \leq \sigma
\end{align}
$$

The physical intuition remains the same: $\nu \Delta t$ is the the root-mean-square distance traveled by a fluid element in time $\Delta t$ because of diffusion and this has to be a constant fraction $\sigma < 1$ of $\Delta x^2$. Below we implement Equation \eqref{eqn:1ddiffusion-discrete} in Python, while omitting the code for intializing variables and setting initial conditions, which has already been shown in Step 1)

```python
for n in range(nt):
    un = u.copy()
    u[1:-1] = un[1:-1] + nu  * dt / dx**2 * (un[2:] - 2 * un[1:-1] + un[:-2])
    for i in range(1, nx - 1):
        u[i] = (un[i] + nu * dt / dx**2 * (un[i+1] - 2 * un[i] + 
                un[i-1]))
```
![test image size](/assets/images/coursenotes/CFDpython/step3-lowdiff.png){: width="49%"} ![test image size](/assets/images/coursenotes/CFDpython/step3-highdiff.png){: width="49%"}

The image on the left was generated with `nt = 20` and the one on the righ with `nt = 400`. We see how diffusion tends to 'smear out' the intial square velocity waveform. 

## Step 4: Burgers' Equation

The Burgers' equation combines 1-D nonlinear convection with 1-D diffusion, i.e., puts together step 2 and step 3 from above. The equation is

$$
\begin{align}
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial ^2u}{\partial x^2}
\end{align}
$$

We can discrete the first order and second order terms as described above and then rearrange terms to solve for the only unknown $u_i^{n+1}$. Putting together everything and carrying out the algebra, we arrive at

$$
\begin{align}
u_i^{n+1} = u_i^n - u_i^n \frac{\Delta t}{\Delta x} (u_i^n - u_{i-1}^n) + \nu \frac{\Delta t}{\Delta x^2}(u_{i+1}^n - 2u_i^n + u_{i-1}^n)
\end{align}
$$

along with the intial and boundary conditions given in Equation \eqref{eqn:ICs}. Because this is a straightforward implementation of steps 2 and 3 in a single equation, we omit any detailed analysis and move on to step 5. 

## Step 5: 2-D Linear Convection

We modify Equation \eqref{eqn:wave} to now include a derivative in the $y$ direction:

$$
\begin{equation}
\dfrac{\partial u}{\partial t}+c\left(\dfrac{\partial u}{\partial x} + \dfrac{\partial u}{\partial y}\right) = 0 \label{eqn:wave2d}
\end{equation}
$$

By now, discretizing equations has become routine! Equation \eqref{eqn:wave2d}, discretized and rearranged, becomes

$$
\begin{equation}
u_{i,j}^{n+1} = u_{i,j}^n-c \frac{\Delta t}{\Delta x}(u_{i,j}^n-u_{i-1,j}^n)-c \frac{\Delta t}{\Delta y}(u_{i,j}^n-u_{i,j-1}^n)
\end{equation}
$$

So far, because we lived in 1-D space we modeled $u_i$ as a being a vector of length $M+1$. While expanding to 2 dimensions, we now model $u_{i,j}$ as being a 2-D array, indexed by indices $i,j$.  The intial and boundary conditions are

$$
\begin{align}
u(x,y) & = \begin{cases}
\begin{matrix}
2,  & 0.5 \leq x, y \leq 1 \cr
1, & \text{otherwise}
\end{matrix}
\end{cases} \label {eqn:IC2d}\\

u(x,y) & = 1\ \text{for } \begin{cases}
\begin{matrix}
x =  0,\ 2 \cr
y =  0,\ 2 \end{matrix}\end{cases} \label{eqn:BC2d}
\end{align}
$$

```python
from mpl_toolkits.mplot3d import Axes3D  # Required for 3d plots

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
%matplotlib inline

# variable initialization
nx = ny = 201
dx = 2 / (nx-1)
dy = 2 / (ny-1)
c = 1
sigma = .5 # from the CLF condition
dt = sigma * (dx / c)
nt = int(.7/dt)
u = np.ones((nx, ny))

# Initial conditions
u[int(0.5/dx):int(1/dx) + 1, int(0.5/dy):int(1/dy) + 1] = 2

# Plot Initial Conditions
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, nx)

fig = plt.figure(figsize=(11,7))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y, sparse=True)
ax.plot_surface(X, Y, u, cmap='viridis')
```

![test image size](/assets/images/coursenotes/CFDpython/step5-initial.png){: width="75%" .align-center}

The inital waveform is a rectangular box. We now implement the numerical scheme and see how this initial waveform is convected and diffused over time in two dimensions.

```python
for i in range(nt+1):
    un = u.copy()
    u[1:,1:] = (un[1:, 1:] - (c*dt/dx)*(un[1:, 1:] - un[:-1, 1:]) -
                (c*dt/dy)*(un[1:, 1:] - un[1:, :-1]))
# Do the 3D plotting of the results
fig_result = plt.figure(figsize=(11,7))
ax = fig_result.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, u, cmap='viridis')
```
![test image size](/assets/images/coursenotes/CFDpython/step5-final.png){: width="75%" .align-center}

As expected the waveform has been translated (convected) as well as smeared-out (diffused) in two dimensions.

## Step 6: 2-D Nonlinear Convection

We now extend our analysis to 2-D nonlinear convection, where have velocity components $u$ in the $x$ direction and $v$ in the $y$ direction. The transport equations are

$$
\begin{align}
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} & = 0 \\
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} & = 0
\end{align}
$$

Note that we have two equation because the original NS equation is a vector equation, with one equation for each component of velocity. Discretizing these equations, and solving for the two unknowns $u_i^{n+1}$ and $v_n^{n+1}$, we obtain

$$
\begin{align}
u_{i,j}^{n+1} = u_{i,j}^n - u_{i,j} \frac{\Delta t}{\Delta x} (u_{i,j}^n-u_{i-1,j}^n) - v_{i,j}^n \frac{\Delta t}{\Delta y} (u_{i,j}^n-u_{i,j-1}^n) \\

v_{i,j}^{n+1} = v_{i,j}^n - u_{i,j} \frac{\Delta t}{\Delta x} (v_{i,j}^n-v_{i-1,j}^n) - v_{i,j}^n \frac{\Delta t}{\Delta y} (v_{i,j}^n-v_{i,j-1}^n)
\end{align}
$$

The intial conditions are the same as \label{eqn:IC2d} but the boundary condition is modified to

$$
\begin{align}
u,\ v\ = \begin{cases}\begin{matrix}
2 & \text{for } x,y \in (0.5, 1)\times(0.5,1) \cr
1 & \text{everywhere else}
\end{matrix}\end{cases} \label{Eqn:BC2dnew}
\end{align}
$$

The solution is implemented as follows:

```python
# Initializations
nx = 101
ny = 101
dx = 2 / (nx-1)
dy = 2 / (ny-1)

sigma = .2
c = 1
dt = sigma * dx / c
nt = int(.5/dt)

# Initial conditions
u = np.ones((nx,ny))
v = np.ones((nx, ny))

u[int(.5/dx):int(1/dx) + 1, int(.5/dy):int(1/dy) + 1] = 2
v[int(.5/dy):int(1/dy) + 1, int(.5/dy):int(1/dy) + 1] = 2
```

A plot of the intial conditions looks the same as in Step 5. We proceed to solving the equations and plotting the results

```python
# Solution
un = u.copy()
vn = v.copy()

for _ in range(nt + 1):
    un = u.copy()
    vn = v.copy()
    u[1:,1:] = un[1:, 1:] - (un[1:, 1:]*dt/dx)*(un[1:, 1:] - un[:-1, 1:]) - (vn[1:, 1:]*dt/dy)*(un[1:, 1:] - un[1:, :-1])
    v[1:,1:] = vn[1:, 1:] - (un[1:, 1:]*dt/dx)*(vn[1:, 1:] - vn[:-1, 1:]) - (vn[1:, 1:]*dt/dy)*(vn[1:, 1:] - vn[1:, :-1])

x = np.linspace(0,2, nx)
y = np.linspace(0,2, ny)

X, Y = np.meshgrid(x, y)

# Plot of u
fig_result_u = plt.figure(figsize=(11,7))
ax_u = fig_result_u.add_subplot(111, projection='3d')
ax_u.plot_surface(X, Y, u, cmap='viridis')
plt.colorbar(v_plot)

# Plot of v
fig_result_v = plt.figure(figsize=(11,7))
ax_v = fig_result_v.add_subplot(111, projection='3d')
v_plot = ax_v.plot_surface(X, Y, v, cmap='viridis')
plt.colorbar(v_plot)
```

![test image size](/assets/images/coursenotes/CFDpython/step6-usoln.png){: width="49%"} ![test image size](/assets/images/coursenotes/CFDpython/step6-vsoln.png){: width="49%"}

## Step 7: 2-D Diffusion

The 2-D diffusion equation is:

$$
\begin{align}
\frac{\partial u}{\partial t} = \nu \frac{\partial ^2 u}{\partial x^2} + \nu \frac{\partial ^2 u}{\partial y^2}
\end{align}
$$

which can be descritized and rearranged to find $u_i^{n+1}:

$$
\begin{align}
u_{i,j}^{n+1} = & u_{i,j}^n + \frac{\nu \Delta t}{\Delta x^2}(u_{i+1,j}^n - 2 u_{i,j}^n + u_{i-1,j}^n) +  \\
& \frac{\nu \Delta t}{\Delta y^2}(u_{i,j+1}^n-2 u_{i,j}^n + u_{i,j-1}^n)
\end{align}
$$

We define a function `diffusion_plotter(nt)` which takes in the number of timesteps to calculate so that we can see how diffusion smears out the intial pulse in two dimensions (intializations and setting up of intial conditons can be implemented as above)

```python 
def diffusion_plotter(nt):
    u = np.ones((nx, ny))
    u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2  
    
    un = u.copy()
    for _ in range(nt + 1):
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] + nu*dt/dx**2*(un[2:,1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]) + 
                         nu*dt/dy**2*(un[1:-1,2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2]))
        
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1
        
        un = u.copy()
        
    fig_result = plt.figure(figsize=(11,7))
    ax_result = fig_result.add_subplot(111)
    result_plot = ax_result.contourf(X, Y, u, 20)
    plt.colorbar(result_plot)
```

![test image size](/assets/images/coursenotes/CFDpython/step7-timestep5.png){: width="49%"} ![test image size](/assets/images/coursenotes/CFDpython/step7-timestep15.png){: width="49%"}
![test image size](/assets/images/coursenotes/CFDpython/step7-timestep50.png){: width="49%"} ![test image size](/assets/images/coursenotes/CFDpython/step7-timestep100.png){: width="49%"}

## Step 8: 2-D Burgers' Equation

The 2-D Burgers' equation puts together 2-D nonlinear convection (Step 6) and 2-D diffusion (Step 7). The equations for $u$ and $v$ are

$$
\begin{align}
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = \nu \; \left(\frac{\partial ^2 u}{\partial x^2} + \frac{\partial ^2 u}{\partial y^2}\right) \\

\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = \nu \; \left(\frac{\partial ^2 v}{\partial x^2} + \frac{\partial ^2 v}{\partial y^2}\right)
\end{align}
$$

There can be discretized and rearranged (after some algebra!) to obtain

$$
\begin{align}
\nonumber u_{i,j}^{n+1} = & u_{i,j}^n - \frac{\Delta t}{\Delta x} u_{i,j}^n (u_{i,j}^n - u_{i-1,j}^n)  - \frac{\Delta t}{\Delta y} v_{i,j}^n (u_{i,j}^n - u_{i,j-1}^n) + \\
& \frac{\nu \Delta t}{\Delta x^2}(u_{i+1,j}^n-2u_{i,j}^n+u_{i-1,j}^n) + \frac{\nu \Delta t}{\Delta y^2} (u_{i,j+1}^n - 2u_{i,j}^n + u_{i,j-1}^n)
\end{align}
$$

$$
\begin{align}
\nonumber v_{i,j}^{n+1} = & v_{i,j}^n - \frac{\Delta t}{\Delta x} u_{i,j}^n (v_{i,j}^n - v_{i-1,j}^n) - \frac{\Delta t}{\Delta y} v_{i,j}^n (v_{i,j}^n - v_{i,j-1}^n) \\
&+ \frac{\nu \Delta t}{\Delta x^2}(v_{i+1,j}^n-2v_{i,j}^n+v_{i-1,j}^n) + \frac{\nu \Delta t}{\Delta y^2} (v_{i,j+1}^n - 2v_{i,j}^n + v_{i,j-1}^n)
\end{align}
$$

We implement the solution below:

```python
# Initialization
nx = 41
ny = 41
nt = 2000
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .0009
nu = 0.01
dt = sigma * dx * dy / nu


x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)

u = numpy.ones((ny, nx))
v = numpy.ones((ny, nx))
un = numpy.ones((ny, nx)) 
vn = numpy.ones((ny, nx))
comb = numpy.ones((ny, nx))

# Initial conditions
u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2 
v[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2

for n in range(nt + 1): ##loop across number of time steps
    un = u.copy()
    vn = v.copy()

    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     dt / dx * un[1:-1, 1:-1] * 
                     (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - 
                     dt / dy * vn[1:-1, 1:-1] * 
                     (un[1:-1, 1:-1] - un[0:-2, 1:-1]) + 
                     nu * dt / dx**2 * 
                     (un[1:-1,2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + 
                     nu * dt / dy**2 * 
                     (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
    
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - 
                     dt / dx * un[1:-1, 1:-1] *
                     (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     dt / dy * vn[1:-1, 1:-1] * 
                    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) + 
                     nu * dt / dx**2 * 
                     (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                     nu * dt / dy**2 *
                     (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
     
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

# Plot the solution
fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u, cmap=cm.viridis, rstride=1, cstride=1)
ax.plot_surface(X, Y, v, cmap=cm.viridis, rstride=1, cstride=1)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$');
```

![test image size](/assets/images/coursenotes/CFDpython/step8-soln.png){: width="100%" .align-center}

The intial pulse has been translated, sheared, and smeared, as expected by the different elements of the Burgers' equation. 

## References

[^1]:Batchelor, G.K., 1967. An introduction to fluid dynamics. Cambridge university press.
[^2]:Courant, R., Friedrichs, K. and Lewy, H., 1967. On the partial difference equations of mathematical physics. IBM journal of Research and Development, 11(2), pp.215-234.


