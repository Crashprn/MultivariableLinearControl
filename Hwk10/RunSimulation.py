import numpy as np
import control as ct
import sympy as sp
from scipy import integrate
import matplotlib.pyplot as plt
from Segway import Segway
import typing as t

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
    def plotResults(self, indexes: t.List[int], formats: t.List[str], titles: t.List[str], title: str, save=True, target=None, couple=False) -> None:
        time = self.timespan.reshape(-1,1)
        data = np.concatenate((self.timespan.reshape(-1, 1), self.last_sim_x, self.last_sim_u), axis=1)
        plot_xy = []
        if couple:
            plot_xy = [(indexes[i], indexes[i+1]) for i in range(0, len(indexes), 2)]
        else:
            plot_xy = [(0, i+1) for i in indexes]

        numplots = len(plot_xy)
        fig, axs = plt.subplots(numplots)

        for i,(x,y) in enumerate(plot_xy):
            x_d = data[:, x]
            y_d = data[:, y]
            if numplots == 1:
                axs.plot(x_d, y_d, formats[i])
                if target is not None:
                    axs.plot(x_d, np.ones(len(x_d))*target[i], 'k--')
                axs.set(xlabel='time', ylabel=titles[i])

            else:
                axs[i].plot(x_d, y_d, formats[i])
                if target is not None:
                    axs[i].plot(x_d, np.ones(len(x_d))*target[i], 'k--')
                axs[i].set(xlabel='time', ylabel=titles[i])
                axs[i].set_xlim([x_d[0], x_d[-1]])

        fig.suptitle(title)

        if save:
            plt.savefig(title.replace(" ", "_") + ".png")


def createDynamics() -> t.Tuple[np.ndarray, np.ndarray, t.List[t.Callable]]:
    # Parameters
    m_c, m_s, d, L, R, I_2, I_3, g = sp.symbols('m_c m_s d L R I_2 I_3 g')
    params = {m_c: .503, m_s: 4.315, d:.1, L:.1, R: .073, I_2: .003679, I_3: .02807, g: 9.8}

    # States
    x, y, v, v_dot, phi, phi_dot, phi_ddot, psi, omega, omega_dot = sp.symbols('x y v v_dot phi phi_dot phi_ddot psi omega omega_dot')

    # Controls
    u_1, u_2 = sp.symbols('u_1 u_2')

    # Setup Equations
    f1 = 3*(m_c+m_s)*v_dot - m_s*d*sp.cos(phi)*phi_ddot + m_s*d*sp.sin(phi)*(phi_dot**2+omega**2) + u_1/R
    f2 = ( (3*L**2 + 1/(2*R**2))*m_c + m_s*d**2*sp.sin(phi)**2 + I_2 )*omega_dot + m_s*d**2*sp.sin(phi)*sp.cos(phi)*omega*phi_dot - L/R*u_2
    f3 = m_s*d*sp.cos(phi)*v_dot + (-m_s*d**2-I_3)*phi_ddot + m_s*d**2*sp.sin(phi)*sp.cos(phi)*phi_dot**2 + m_s*g*d*sp.sin(phi) - u_1

    # Solve Equations
    soln = sp.solve((f1, f2, f3), (v_dot, phi_ddot, omega_dot), dict=True)[0]
    phi_ddot = soln[phi_ddot]
    omega_dot = soln[omega_dot]
    v_dot = soln[v_dot]

    x1_dot_f = sp.lambdify((v, psi), v*sp.cos(psi), 'numpy')
    x2_dot_f = sp.lambdify((v, psi), v*sp.sin(psi), 'numpy')
    omega_dot_f = sp.lambdify((omega, phi, phi_dot, u_2), omega_dot.subs(params), 'numpy')
    v_dot_f = sp.lambdify((omega, phi, phi_dot, u_1), v_dot.subs(params), 'numpy')
    phi_ddot_f = sp.lambdify((omega, phi, phi_dot, u_1), phi_ddot.subs(params), 'numpy')

    dynamics = [x1_dot_f, x2_dot_f, omega_dot_f, v_dot_f, phi_ddot_f]

    # Create A and B matrices for subsystem z at eq = 0
    Z = sp.Matrix([omega, v, phi, phi_dot])
    U = sp.Matrix([u_1, u_2])
    f = sp.Matrix([[omega_dot], [v_dot], [phi_dot], [phi_ddot]])

    equillibrium = {omega:0, v:0, phi:0, phi_dot:0, u_1:0, u_2:0}

    df_dz = f.jacobian(Z)
    df_du = f.jacobian(U)

    A = df_dz.subs(equillibrium).subs(params)
    B = df_du.subs(equillibrium).subs(params)

    A = np.array(A).astype(np.float64)
    B = np.array(B).astype(np.float64)
    
    return (A,B,dynamics)


def Simulate():
    
    # Simulation Part
    x0 = np.array([0, 0, 0, 0, .5, np.pi/4, 0])
    v_d = 1
    omega_d = 0.25
    A, B, dynamics = createDynamics()
    segway = Segway(v_d, omega_d, A, B, dynamics, [-2, -2, -3, -4])

    t0 = 0
    tf = 20
    dt = 0.01

    simulation = Simulation(t0, dt, tf, 2)

    simulation.pythonODE(segway.x_dot, x0)
    simulation.getControlVector(segway.get_feedback_control)

    #simulation.plotResults(np.array([3, 4, 5, 6, 7, 8]), ['b', 'b', 'b', 'b', 'r', 'r'], ['Omega', 'V', 'Phi', 'Phi_dot', 'U_1', 'U_2'], "State Plot", save=False)

    simulation.plotResults(np.array([1, 2]), ['b', 'b'], ['X', 'Y'], "X and Y plot", save=False, couple=True)
    simulation.plotResults(np.array([5, 4, 3]), ['b', 'b', 'b', 'b', 'r', 'r'], ['Tilt Angle', 'Trans. Vel', 'Rot. Vel'], "State Plot",
                           target=[0, v_d, omega_d], save=False)

    plt.show()




if __name__ == "__main__":
    Simulate()
    





