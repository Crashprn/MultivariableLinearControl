import numpy as np
import scipy as sp
import typing as t
import scipy.integrate as integrate
import matplotlib.pyplot as plt

class Simulation:

    def __init__(self, t0, dt, tf, u_size) -> None:
        self.dt: float = dt
        self.timespan: np.ndarray = np.arange(t0, tf, dt)
        self.u_size: int = u_size
        self.last_sim_x = []
        self.last_sim_u = []
        self.sims = []
    
    '''
    Calculates the control vector over the specified time span

    INPUTS:
        tvec: 1xm vector of time inputs
        xvec: mxn matrix of states
        u: function for control vector

    OUTPUTS:
        uvec: mxn vector of control inputs
    '''
    def getControlVector(self, u) -> np.ndarray:
        u_vec = np.zeros((len(self.timespan), self.u_size))
        for i in range(len(self.timespan)):
            u_vec[i,:] = u(self.timespan[i], self.last_sim_x[i, :]).squeeze()    
        self.last_sim_u = u_vec

        return u_vec
    
    '''
        Uses ODEint from the scipy.integrate.odeint function to solve the ODEs

        INPUTS:
            x0: 1xn vector of initial states
            u: function for control vector
        OUTPUTS:
            xvec: mxn matrix of states
    '''
    def pythonODE(self, f, x0) -> t.Tuple[np.ndarray, np.ndarray]:
        xvec = integrate.odeint(f, x0, self.timespan, tfirst=True)
        self.last_sim_x = xvec
    
    def get_last_sim_x(self) -> np.ndarray:
        return self.last_sim_x
    
    def save_last_sim(self) -> None:
        sim = np.concatenate((self.timespan.reshape(-1, 1), self.last_sim_x, self.last_sim_u), axis=1)
        self.sims.append(sim)
    
    '''
        Plots the results of the simulation
        INPUTS:
            data: list of tuples of time, state vectors
            Titles: list of titles for each plot
    '''
    def plotResults(self,
                    indexes: t.List[int],
                    formats: t.List[str],
                    titles: t.List[str], 
                    title: str, 
                    save=True, 
                    target=None, 
                    couple=False, 
                    all_sims=True,
                    sim_names=['Sim']) -> None:
        if all_sims:
            runs = self.sims
        else:
            runs = [self.sims[-1]]


        plot_xy = []
        if couple:
            plot_xy = [(indexes[i], indexes[i+1]) for i in range(0, len(indexes), 2)]
        else:
            plot_xy = [(0, i+1) for i in indexes]

        numplots = len(plot_xy)
        fig, axs = plt.subplots(numplots)
        
        for sim_name, format, data in zip(sim_names, formats, runs):
            for i,(x,y) in enumerate(plot_xy):
                x_d = data[:, x]
                y_d = data[:, y]
                if numplots == 1:
                    axs.plot(x_d, y_d, format, label=sim_name)
                    if target is not None:
                        axs.plot(x_d, np.ones(len(x_d))*target[i], 'k--')
                    axs.set(xlabel=titles[i][0], ylabel=titles[i][1])

                else:
                    axs[i].plot(x_d, y_d, format, label=sim_name)
                    if target is not None:
                        axs[i].plot(x_d, np.ones(len(x_d))*target[i], 'k--')
                    axs[i].set(xlabel='time', ylabel=titles[i])
                    axs[i].set_xlim([x_d[0], x_d[-1]])
                
                if all_sims:
                    axs[i].legend()

        fig.suptitle(title)

        if save:
            plt.savefig(title.replace(" ", "_") + ".png")

