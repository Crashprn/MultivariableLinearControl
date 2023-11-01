import numpy as np
import typing as t
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sympy as sp
import control as ct

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
        u_vec = np.zeros((len(self.timespan),1))
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


def Simulate(zero_state_value):
    t0 = 0
    dt = 0.01
    tf = 10.0
    u = lambda t, x: np.array([np.sin(t)])
    x0 = np.array([0, 0, np.pi - .25, 0])

    x, x_dot, x_ddot, theta, theta_dot, theta_ddot, u = sp.symbols('x x_dot x_ddot theta theta_dot theta_ddot u')
    M, m, b, I, g, l = sp.symbols('M m b I g l')
    params = {M: .5, m: .2, b: 0.1, I: 0.006, g: 9.8, l: 0.3}

    f1 = (M+m)*x_ddot + b*x_dot + m*l*theta_ddot*sp.cos(theta) - m*l*theta_dot**2*sp.sin(theta) - u
    f2 = (I + m*l**2)*theta_ddot + m*g*l*sp.sin(theta) + m*l*x_ddot*sp.cos(theta)

    soln = sp.solve((f1, f2), (x_ddot, theta_ddot), dict=True)[0]
    x_ddot = soln[x_ddot]
    theta_ddot = soln[theta_ddot]

    x_ddot_f = sp.lambdify((x, x_dot, theta, theta_dot, u), x_ddot.subs(params), 'numpy')
    theta_ddot_f = sp.lambdify((x, x_dot, theta, theta_dot, u), theta_ddot.subs(params), 'numpy')

    X = sp.Matrix([x, x_dot, theta, theta_dot])
    U = sp.Matrix([u])

    f = sp.Matrix([[x_dot], [x_ddot], [theta_dot], [theta_ddot]])

    zero_state = {x: 0, x_dot: 0, theta: zero_state_value, theta_dot: 0, u: 0}

    df_dx = f.jacobian(X).subs(params)
    df_du = f.jacobian(U).subs(params)


    # Linearize
    A = df_dx.subs(zero_state)
    B = df_du.subs(zero_state)

    sp.pprint(A)
    sp.pprint(B)

    # Evaluating Stability
    A = np.array(A).astype(np.float64)
    B = np.array(B).astype(np.float64)

    eigs = np.linalg.eig(A)[0]
    print("Eignevalues: ", end="")
    print(*eigs, sep=", ")

    # Controllability
    gamma = ct.ctrb(A, B)
    print("Rank of Controllability Matrix: ", np.linalg.matrix_rank(gamma))
    K = ct.place(A, B, [-1, -2, -3, -4])
    print("K: ")
    sp.pprint(sp.Matrix(K))

    k_eigs = np.linalg.eig(A - B @ K)[0]
    print("Eignevalues of A - BK: ", end="")
    print(*k_eigs, sep=", ")

    # Simulate

    def u_func (t, x):
        x_eq = np.array([0, 0, zero_state_value, 0])
        return -K @ (x-x_eq).T  + zero_state[u]
    '''
        Calculates the state dynamics using the current time, state, and control vector
    '''
    def f(t, x_in, u_in) -> np.ndarray:
        uvec = u_in(t, x_in)

        thetaddot = theta_ddot_f(x_in[0], x_in[1], x_in[2], x_in[3], uvec[0])
        xddot = x_ddot_f(x_in[0], x_in[1], x_in[2], x_in[3], uvec[0])


        xdot  = np.array([x_in[1], xddot, x_in[3], thetaddot])
        return xdot



    sim = Simulation(t0, dt, tf, u, f, x0)

    data = []

    tvec, xvec = sim.pythonODE(x0, u_func)
    uvec = sim.getControlVector(xvec, u_func)
    data.append((tvec, xvec, uvec))

    plotResults(data, ['b', 'r:'])

if __name__ == "__main__":
    Simulate(np.pi)
    plt.show()