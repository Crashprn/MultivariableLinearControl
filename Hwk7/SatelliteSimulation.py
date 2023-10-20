import numpy as np
from scipy import integrate
import sympy as sp
import control as ct
import typing as t
import matplotlib.pyplot as plt

class Simulation:

    def __init__(self, t0, dt, tf, u, f, x0) -> None:
        self.dt: float = dt
        self.u: t.Callable = u
        self.x0: np.ndarray = x0
        self.f: t.Callable = f
        self.timespan: np.ndarray = np.arange(t0, tf, dt)
    
    '''
    Calculates the control vector over the specified time span

    INPUTS:
        tvec: 1xm vector of time inputs
        xvec: mxn matrix of states
        u: function for control vector

    OUTPUTS:
        uvec: mxn vector of control inputs
    '''
    def getControlVector(self, xvec, u, u_sol) -> np.ndarray:
        u_vec = np.zeros((len(self.timespan),2))
        for i in range(len(self.timespan)):
            u_vec[i,:] = u(self.timespan[i], xvec[i, :], u_sol).squeeze()
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
        xvec = integrate.odeint(self.f, x0, self.timespan, args=(u,), tfirst=True)
        return self.timespan, xvec

def controllability(A, B):
    n = A.shape[0]
    eigs = A.eigenvals().keys()
    for eig in eigs:
        Control = sp.Matrix([(eig * sp.eye(n) - A).T, B.T]).T
        print('Rank of Augmented Eig {}: {}'.format(eig, Control.rank()))

def SatelliteSimulation():
    ## Initialize the simulation variables
    # Parameters
    params = {'mu': 0.004302, 'R': 200}
    # Initialize state variables
    r0 = 205
    rdot0 = 0.0
    theta0 = 0
    thetadot0 = 0
    x0 = np.array([r0, theta0, rdot0, thetadot0])
    
    # Initialize sim variables
    t0 = 0 # initial time
    dt = 0.01 # time step
    tf = 10.0 # final time    
    
    # Calculate the feedback matrix
    x_1, x_2, x_3, x_4, u_1, u_2 = sp.symbols('x_1, x_2, x_3, x_4, u_1, u_2')
    R, mu, t = sp.symbols('R, mu, t')

    A = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [x_4**2 + 2*(mu/x_1**3), 0, 0, 2*x_4*x_1], [2*(x_3*x_4/x_1**2) + u_1/x_1, 0, -2*x_4/x_1, -2 * x_3/x_1]])
    B = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1/params['R']]])
    sub = {x_1: R, x_2: sp.sqrt(mu/R**3)*t, x_3: 0, x_4: sp.sqrt(mu/R**3), u_1: 0, u_2: 0}
    A_1 = sp.simplify(A.subs(sub))
    sp.pprint(A_1)
    A_2 = A_1.subs(params)
    # 2 and 3)
    controllability(A_1, B)

    # 4)
    A_2 = np.array(A_2.tolist()).astype(np.float64)
    B_2 = np.array(B.tolist()).astype(np.float64)

    K = ct.place(A_2, B_2, [-1, -1, -2, -2])
    print('Feedback matrix: ')
    print(K)

    print('Eigenvalues of A - BK: ')
    print(*np.linalg.eig(A_2 - B_2@K)[0], sep=', ')

    def getXSol(t):
        return np.array([params['R'], np.sqrt(params['mu']/params['R']**3)*t, 0, np.sqrt(params['mu']/params['R']**3)])

    def input(t, x, x_sol):
        delta_x = x - x_sol(t)
        u = -K@(delta_x.T)
        return u.T
    # Set the control input function

    u = input 
    
    
    ## Simulate and plot the system using ode
    # Simulate the system
    def f(t, x, u) -> np.ndarray:
        # Extract states
        r = x[0] # Radius
        theta = x[1] # Orbit angle
        rdot = x[2] # Time derivative of radius
        thetadot = x[3] # Time derivative of orbit angle
        
        # Extract control inputs
        ar, ai = u(t, x, getXSol)
        
        # Calculate the second derivatives
        r_ddot = thetadot**2*r - params['mu']/(r**2) + ar
        theta_ddot = 1/r*(ai - 2*rdot*thetadot)
        
        # Output the dynamics
        xdot = np.array([rdot, thetadot, r_ddot, theta_ddot])
        return xdot


    sim = Simulation(t0, dt, tf, u, f, x0)

    data = []

    tvec, xvec = sim.pythonODE(x0, u)
    uvec = sim.getControlVector(xvec, u, getXSol)
    data.append((tvec, xvec, uvec))

    plotResults(data, ['b', 'r:'], ['r', 'theta', 'rdot', 'thetadot', 'a_i', 'a_r'])   


def plotResults(data: t.List[t.Tuple[np.ndarray, np.ndarray, np.ndarray]], formats: t.List[str], x_titles: t.List[str]) -> None:
    numx = data[0][1].shape[1] 
    numu =  1 if len(data[0][2].shape) <= 1 else data[0][2].shape[1]
    numplots = numx + numu
    fig, axs = plt.subplots(numplots)
    for i, dat in enumerate(data):
        tvec, xvec, uvec = dat
        for j in range(numx):
            # Plotting x's on subplot 
            axs[j].plot(tvec, xvec[:, j], formats[i])   
            axs[j].set(xlabel='time', ylabel=x_titles[j])
        
        for j in range(numx, numplots):
            # Plotting u's on subplot
            axs[j].plot(tvec, uvec[:, j-numx], formats[i])
            axs[j].set(xlabel='time', ylabel=x_titles[j])
    plt.show()


if __name__ == '__main__':
    SatelliteSimulation()


