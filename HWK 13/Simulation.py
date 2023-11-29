import numpy as np
from scipy.integrate import odeint
import scipy
import typing as t
import matplotlib.pyplot as plt

class Simulation:

    def __init__(self, t0, dt, tf, u, f, x0, u_dim=1) -> None:
        self.dt: float = dt
        self.u: t.Callable = u
        self.x0: np.ndarray = x0
        self.f: t.Callable = f
        self.timespan: np.ndarray = np.arange(t0, tf, dt)
        self.u_dim = u_dim
    
    '''
    Calculates the control vector over the specified time span

    INPUTS:
        tvec: 1xm vector of time inputs
        xvec: mxn matrix of states
        u: function for control vector

    OUTPUTS:
        uvec: mxn vector of control inputs
    '''
    def getControlVector(self, xvec, u) -> np.ndarray:
        u_vec = np.zeros((len(self.timespan), self.u_dim))
        for i in range(len(self.timespan)):
            u_vec[i,:] = u(self.timespan[i], xvec[i, :]).squeeze()
        return u_vec
    
    '''
        Uses ODEint from the scipy.integrate.odeint function to solve the ODEs

        INPUTS:
            x0: 1xn vector of initial states
            u: function for control vector
        OUTPUTS:
            xvec: mxn matrix of states
    '''
    def pythonODE(self, x0, u) -> t.Tuple[np.ndarray, np.ndarray]:
        xvec = odeint(self.f, x0, self.timespan, args=(u,), tfirst=True)
        return self.timespan, xvec
    
    '''
        Uses euler integration to solve the ODEs

        INPUTS:
            x0: 1xn vector of initial states
            u: function for control vector
        OUTPUTS:
            tvec: 1xm vector of time inputs
            xvec: mxn matrix of states
    '''
    def eulerIntegration(self, x0, u) -> t.Tuple[np.ndarray, np.ndarray]:
        xvec = np.zeros((len(self.timespan), len(x0)))
        xvec[0, :] = x0
        for i in range(1, len(self.timespan)):
            t = self.timespan[i]
            x = xvec[i-1, :]
            xvec[i, :] = xvec[i-1, :] + self.dt*self.f(t, x, u)
        return self.timespan, xvec

    

'''
    Plots the results of the simulation
    INPUTS:
        data: list of tuples of time, state vectors
        Titles: list of titles for each plot
'''
def plotResults(data: t.List[t.Tuple[np.ndarray, np.ndarray, np.ndarray]], formats: t.List[str], titles: t.List[str], title: str, save=True, legend=True) -> None:
    numx = data[0][1].shape[1] 
    numu =  1 if len(data[0][2].shape) <= 1 else data[0][2].shape[1]
    numplots = numx + numu
    leg = ['Odeint', 'Euler']
    fig, axs = plt.subplots(numplots)
    for i, dat in enumerate(data):
        tvec, xvec, uvec = dat
        for j in range(numx):
            # Plotting x's on subplot 
            axs[j].plot(tvec, xvec[:, j], formats[i], label=leg[i])   
            axs[j].set(xlabel='time', ylabel=titles[j])
            if legend:
                axs[j].legend()
        
        for j in range(numx, numplots):
            # Plotting u's on subplot
            axs[j].plot(tvec, uvec[:, j-numx], formats[i])
            axs[j].set(xlabel='time', ylabel=f'u{j-numx + 1}')
    
    fig.suptitle(title)

    if save:
        plt.savefig(title.replace(" ", "_") + ".png")