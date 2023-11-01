import control as ct
import numpy as np
import typing as t
import scipy as sy
import matplotlib.pyplot as plt
import sympy as sp

from scipy import integrate
from scipy import linalg

class Simulation:

    def __init__(self, t0, dt, tf, u_size) -> None:
        self.dt: float = dt
        self.timespan: np.ndarray = np.arange(t0, tf, dt)
        self.u_size: int = u_size
    
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
        u_vec = np.zeros((len(self.timespan), self.u_size))
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
    def pythonODE(self, f, u, x0) -> t.Tuple[np.ndarray, np.ndarray]:
        xvec = integrate.odeint(f, x0, self.timespan, args=(u,), tfirst=True)
        return self.timespan, xvec
    
'''
    Plots the results of the simulation
    INPUTS:
        data: list of tuples of time, state vectors
        Titles: list of titles for each plot
'''
def plotResults(data: t.List[t.Tuple[np.ndarray, np.ndarray, np.ndarray]], formats: t.List[str], titles: t.List[str], title: str, save=True, target=None) -> None:
    numx = data[0][1].shape[1] 
    numu =  1 if len(data[0][2].shape) <= 1 else data[0][2].shape[1]
    numplots = numx + numu
    fig, axs = plt.subplots(numplots)
    for i, dat in enumerate(data):
        tvec, xvec, uvec = dat
        for j in range(numx):
            # Plotting x's on subplot 
            axs[j].plot(tvec, xvec[:, j], formats[i])
            if target is not None:
                axs[j].plot(tvec, np.ones(len(tvec))*target, 'k--')
            axs[j].set(xlabel='time', ylabel=titles[j])
            axs[j].set_xlim([tvec[0], tvec[-1]])
        
        for j in range(numx, numplots):
            # Plotting u's on subplot
            axs[j].plot(tvec, uvec[:, j-numx], formats[i])
            axs[j].set(xlabel='time', ylabel=f'u{j-numx + 1}')
            axs[j].set_xlim([tvec[0], tvec[-1]])

    fig.suptitle(title)

    if save:
        plt.savefig(title.replace(" ", "_") + ".png")


def simulate(x1: np.ndarray):
    ## Create the controller
    # Define open loop control parameters
    x2 = 0.5*np.ones(8)
    t1 = 2.5 # Time to get to zero
    t2 = 2.5 # Time to get to x2
    
    # Calculate the control
    A, B, C = getSystemMatrices()
        
    # Set the control input function
    # Finding reachability grammian
    y_0 = np.zeros(8*8)
    def f(t, x):
        y = linalg.expm(A*((t1+t2)-t))@B@(B.T)@(linalg.expm(A.T*((t1+t2)-t)))
        return y.reshape(8*8)
    W_reach = integrate.odeint(f, y_0, np.arange(0, 5.01, 0.01), tfirst=True)[-1].reshape((8,8))

    eta = np.linalg.inv(W_reach)@((x2.T)-linalg.expm(5*A)@(x1.T))

    def u(t, x):
        return (B.T)@(linalg.expm(A.T*(5-t)))@eta
    
    ## Initialize the simulation variables
    # Time variables
    t0 = 0 # initial time
    dt = 0.01 # time step
    tf = t1+t2 #final time
    t = np.arange(t0, tf, dt)
    
    # set initial conditions:
    x0 = x1
    #x0 = rand(8,1);
    def f(t, x, u):
        #f calculates the state dynamics using the current time, state, and
        #control input
        #
        # Inputs:
        #   t: current time
        #   x: current state
        #   u: control input function
        #
        # Ouputs:
        #   xdot: time derivative of x(t)

        u_vec = u(t, x)
        # LTI equation for state
        xdot = A@(x.T) + B@(u_vec)
        return xdot
    
    ## Simulate and plot the system using ode
    # Simulate the system          
    sim = Simulation(t0, dt, tf, 4)
    tvec, xvec = sim.pythonODE(f, u, x0)
    
    uvec = sim.getControlVector(xvec, u)

    data = [(tvec, xvec, uvec)]

    # Plot the resulting states
    plotResults(data, ['b', 'r:'], ['theta', 'phi', 'p', 'q', 'xi', 'v_x', 'v_y', 'v_z'], 'Linear Helicopter Exact Control', save=False, target=0.5)

def getSystemMatrices():
    #Rationalized Helicopter model
    #Code adapted from: http://folk.ntnu.no/skoge/book/2nd_edition/matlab_m/Sec13_2.m
    
    ## Create system matrices 
    # State matrix
    a01 = np.array([ 
                     [0,                  0,                  0,   0.99857378005981],
                     [0,                  0,   1.00000000000000,  -0.00318221934140],
                     [0,                  0, -11.57049560546880,  -2.54463768005371],
                     [0,                  0,   0.43935656547546,  -1.99818229675293],
                     [0,                  0,  -2.04089546203613,  -0.45899915695190],
    [-32.10360717773440,                  0,  -0.50335502624512,   2.29785919189453],
    [  0.10216116905212,  32.05783081054690,  -2.34721755981445,  -0.50361156463623],
    [ -1.91097259521484,   1.71382904052734,  -0.00400543212891,  -0.05741119384766]
    ])
    
    a02 = np.array([
    [ 0.05338427424431,                  0,                  0,                  0],
    [ 0.05952465534210,                  0,                  0,                  0],
    [-0.06360262632370,   0.10678052902222,  -0.09491866827011,   0.00710757449269],
    [                0,   0.01665188372135,   0.01846204698086,  -0.00118747074157],
    [-0.73502779006958,   0.01925575733185,  -0.00459562242031,   0.00212036073208],
    [                0,  -0.02121581137180,  -0.02116791903973,   0.01581159234047],
    [ 0.83494758605957,   0.02122657001019,  -0.03787973523140,   0.00035400385968],
    [                0,   0.01398963481188,  -0.00090675335377,  -0.29051351547241]
    ])

    A=np.concatenate((a01, a02), axis=1)

    # Input matrix
    B=np.array([
    [                 0,                  0,                  0,                  0],
    [                 0,                  0,                  0,                  0],
    [  0.12433505058289,   0.08278584480286,  -2.75247764587402,  -0.01788876950741],
    [ -0.03635892271996,   0.47509527206421,   0.01429074257612,                  0],
    [  0.30449151992798,   0.01495801657438,  -0.49651837348938,  -0.20674192905426],
    [  0.28773546218872,  -0.54450607299805,  -0.01637935638428,                  0],
    [ -0.01907348632812,   0.01636743545532,  -0.54453611373901,   0.23484230041504],
    [ -4.82063293457031,  -0.00038146972656,                  0,                  0]
    ])

    # Output matrix
    C = np.array([
        [ 1, 0, 0, 0, 0, 0, 0, 0], # Pitch
        [ 0, 1, 0, 0, 0, 0, 0, 0], # Roll
        [ 0, 0, 0, 0, 1, 0, 0, 0], # Heading velocity (yaw rate)
        [ 0, 0, 0, 0, 0, 1, 0, 0]  # Heave velocity (forward velocity)
    ])
    return A, B, C


if __name__ == "__main__":
    simulate(np.zeros(8))
    simulate(np.random.rand(8))
    plt.show()