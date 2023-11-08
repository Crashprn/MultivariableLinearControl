import numpy as np
import control as ct
import sympy as sp
from scipy import integrate
import matplotlib.pyplot as plt
from Segway import Segway
import typing as t

class PointControlRobot:
    def __init__(self, A, B, q_d, q_ddot_d, epsilon= 0.1, poles=[-1, -2, -3, -4]) -> None:
        self.A = A
        self.B = B
        self.K = ct.place(A, B, poles)
        self.q_d = q_d
        self.q_ddot_d = q_ddot_d
        self.epsilon = epsilon

        self.x1_ind = 0
        self.x2_ind = 1
        self.psi_ind = 2
        self.v_ind = 3
        self.omega_ind = 4
    
    def get_control(self, t, x) -> float:
        q_d = self.q_d(t)
        q_ddot_d = self.q_ddot_d(t)

        x1 = x[self.x1_ind]
        x2 = x[self.x2_ind]
        psi = x[self.psi_ind]
        v = x[self.v_ind]
        omega = x[self.omega_ind]

        omega_epsilon = np.array([
            [0, self.epsilon*omega],
            [omega/self.epsilon, 0]
        ])

        inv_help = np.array([
            [1, 0],
            [0, 1/self.epsilon]
        ])

        R_epsilon_inv = inv_help@np.array([
            [np.cos(psi), np.sin(psi)],
            [-np.sin(psi), np.cos(psi)]
        ])

        R_epsilon = np.array([
            [np.cos(psi), -self.epsilon*np.sin(psi)],
            [np.sin(psi), self.epsilon*np.cos(psi)]
        ])


        y_x = np.array([
            [x1 + self.epsilon*np.cos(psi)],
            [x2 + self.epsilon*np.sin(psi)]
        ])
        
        y_dot = R_epsilon@np.array([[v], [omega]])

        y = np.concatenate((y_x, y_dot), axis=0)

        q_d = self.q_d(t)

        u = -self.K@(y - q_d) + self.q_ddot_d(t)

        u_total = R_epsilon_inv@u - omega_epsilon@np.array([[v], [omega]])

        return np.concatenate((u_total.T, y_x.T, q_d[0:2,:].T), axis=1).squeeze()

    def x_dot(self, t, x) -> np.ndarray:
        u1, u2, x1_eps, x2_eps, q_d1, q_d2= self.get_control(t, x)

        xdot = np.zeros(x.shape)
        xdot[self.x1_ind] = x[self.v_ind]*np.cos(x[self.psi_ind])
        xdot[self.x2_ind] = x[self.v_ind]*np.sin(x[self.psi_ind])
        xdot[self.psi_ind] = x[self.omega_ind]
        xdot[self.v_ind] = u1
        xdot[self.omega_ind] = u2

        return xdot
        

class Simulation:

    def __init__(self, t0, dt, tf, u_size) -> None:
        self.dt: float = dt
        self.timespan: np.ndarray = np.arange(t0, tf, dt)
        self.u_size: int = u_size
        self.last_sim_x = []
        self.last_sim_u = []
    
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
    
    '''
        Plots the results of the simulation
        INPUTS:
            data: list of tuples of time, state vectors
            Titles: list of titles for each plot
    '''
    def plotResults(self, indexes: t.List[int], formats: t.List[str], titles: t.List[str], title: str, save=True, labels=None, target=None, couple=False, overlay_plot_num=None) -> None:
        data = np.concatenate((self.timespan.reshape(-1, 1), self.last_sim_x, self.last_sim_u), axis=1)
        plot_xy = []
        if couple:
            plot_xy = np.array([[indexes[i], indexes[i+1]] for i in range(0, len(indexes), 2)])
        else:
            plot_xy = np.array([[0, i] for i in indexes])

        if overlay_plot_num is not None:
            plot_xy = plot_xy.reshape(-1, overlay_plot_num, 2)
        else:
            plot_xy = np.expand_dims(plot_xy, axis=0)

        numplots = plot_xy.shape[1]
        fig, axs = plt.subplots(numplots)
        for i, x_y in enumerate(plot_xy):
            for j,(x,y) in enumerate(x_y):
                x_d = data[:, x]
                y_d = data[:, y]
                if numplots == 1:
                    axs.plot(x_d, y_d, formats[i])
                    if target is not None:
                        axs.plot(x_d, np.ones(len(x_d))*target[j], 'k--')
                    axs.set(xlabel=titles[j][0], ylabel=titles[j][1])

                else:
                    axs[j].plot(x_d, y_d, formats[i])
                    if target is not None:
                        axs[j].plot(x_d, np.ones(len(x_d))*target[i], 'k--')
                    axs[j].set(xlabel=titles[j][0], ylabel=titles[j][1])
                    axs[j].set_xlim([x_d[0], x_d[-1]])
        
        if numplots == 1:
            axs.legend()
        else:
            for ax in axs:
                ax.legend(labels=labels)

        fig.suptitle(title)

        if save:
            plt.savefig(title.replace(" ", "_") + ".png")


def Simulate():
    t0 = 0
    dt = 0.01
    tf = 20

    x0 = np.array([0, 0, 0, 0, 0])

    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    B = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    def q_d(t):
        return np.array([
            [np.sin(t)],
            [t],
            [np.cos(t)],
            [1]
        ])
    
    def q_ddot_d(t):
        return np.array([
            [-np.sin(t)],
            [0]
        ])
    
    robot = PointControlRobot(A, B, q_d, q_ddot_d, epsilon=.2, poles=[-3, -4, -5, -6])

    sim = Simulation(t0, dt, tf, 6)

    sim.pythonODE(robot.x_dot, x0)
    sim.getControlVector(robot.get_control)

    sim.plotResults([1, 2, 8, 9, 10, 11], 
                    ['b', 'r', 'g--'], 
                    [('time', 'x'), ('time', 'y')], 
                    'Position',
                    labels=['x', 'x_epsilon', 'x_desired'],
                    save=False, 
                    overlay_plot_num=2
                    )

    plt.show()


if __name__ == "__main__":
    Simulate()