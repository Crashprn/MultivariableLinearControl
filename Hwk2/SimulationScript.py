import numpy as np
import typing as t
import scipy.integrate as integrate
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
    def getControlVector(self, xvec, u) -> np.ndarray:
        u_vec = np.zeros((len(tvec),1))
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
        xvec = integrate.odeint(self.f, x0, self.timespan, args=(u,), tfirst=True)
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
def plotResults(data: t.List[t.Tuple[np.ndarray, np.ndarray, np.ndarray]], formats: t.List[str] ) -> None:
    numx = data[0][1].shape[1] 
    numu =  1 if len(data[0][2].shape) <= 1 else data[0][2].shape[1]
    numplots = numx + numu
    fig, axs = plt.subplots(numplots)
    for i, dat in enumerate(data):
        tvec, xvec, uvec = dat
        for j in range(numx):
            # Plotting x's on subplot 
            axs[j].plot(tvec, xvec[:, j], formats[i])   
            axs[j].set(xlabel='time', ylabel=f'x{j+1}')
        
        for j in range(numx, numplots):
            # Plotting u's on subplot
            axs[j].plot(tvec, uvec[:, j-numx], formats[i])
            axs[j].set(xlabel='time', ylabel=f'u{j-numx + 1}')
    plt.show()


if __name__ == "__main__":
    t0 = 0
    dt = 0.01
    tf = 10.0
    g = 9.8
    m = 1/9.8
    l = .25
    b = 1

    u = lambda t, x: np.array([l/np.sqrt(2)])
    A = np.array([[0, 1], [g/l/np.sqrt(2), -b/(m*l**2)]])
    eigval, eigvec = np.linalg.eig(A)
    #x0 = eigvec[:,1].transpose()
    x0 = np.array([np.pi/4 -.05, 0])

    linearized_x = np.array([np.pi/4, 0])

    linearized_u = np.array([l/np.sqrt(2)])
    '''
        Calculates the state dynamics using the current time, state, and control vector
    '''
    def f(t, x, u) -> np.ndarray:
        B = np.array([[0], [1/(m*l**2)]])
        uvec = u(t, x)
        delta_x = np.transpose(x - linearized_x)
        delta_u = np.transpose(uvec - linearized_u)
        first = A@delta_x
        second = B@delta_u
        xdot = first + second
        ret = xdot.transpose()
        return xdot.transpose()


    sim = Simulation(t0, dt, tf, u, f, x0)

    data = []

    tvec, xvec = sim.pythonODE(x0, u)
    uvec = sim.getControlVector(xvec, u)
    data.append((tvec, xvec, uvec))

    plotResults(data, ['b', 'r:'])