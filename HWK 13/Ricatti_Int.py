import numpy as np
from scipy.integrate import odeint
import scipy
import typing as t
import matplotlib.pyplot as plt
from Simulation import Simulation
from Simulation import plotResults

def plot_result(data, t):
    labels = ['P_11', 'P_12', 'P_12', 'P_22']
    generated_label = ['Regular', 'Numeric']
    style = ['b', 'r--']
    fig, ax = plt.subplots(4, 1, figsize=(10, 10))
    for i, dat in enumerate(data):
        for j in range(dat.shape[1]):
            ax[j].plot(t, dat[:, j], style[i], label=generated_label[i])
            ax[j].set_xlabel('Time (s)')
            ax[j].set_title(labels[j])
            ax[j].legend()


def plot_diff(data, t):
    labels = ['P_11 Error', 'P_12 Error', 'P_12 Error', 'P_22 Error']
    fig, ax = plt.subplots(4, 1, figsize=(10, 10))

    for i in range(data.shape[1]):
        ax[i].plot(t, data[:, i])
        ax[i].set_xlabel('Time (s)')
        ax[i].set_title(labels[i])


def findP(A, B, Q, R, S, t):
    def f(x, t):
        P = np.reshape(x, (2, 2))
        P_dot = -A.T @ P - P @ A - Q + P @ B @ B.T @ P 

        return P_dot.flatten()
    
    P_regular = odeint(f, S.flatten(), t)

    def f(x, t):
        extract = np.reshape(x, (4, 2))
        X = extract[:2, :]
        Y = extract[2:, :]

        X_dot = A@X - B@B.T@Y
        Y_dot = -Q@X - A.T@Y

        return np.concatenate([X_dot, Y_dot]).flatten()

    x_0 = np.concatenate([np.eye(2), S]).flatten()
    x_numeric = odeint(f, x_0, t).reshape(-1, 4, 2)
    X = x_numeric[:, :2, :]
    Y = x_numeric[:, 2:, :]
    P_numeric = np.zeros((len(t), 2, 2))
    for i, (x,y) in enumerate(zip(X, Y)):
        P_numeric[i] = y @ np.linalg.inv(x)
    P_numeric = P_numeric.reshape(-1, 4)

    plot_result([P_regular, P_numeric], t)
    plot_diff(P_regular - P_numeric, t)



def simulate(A, B, Q, R, S, x_0, t_0, dt, T):
    matrix1 = np.concatenate([A, -B@R@B.T], axis=1)
    matrix2 = np.concatenate([-Q, -A.T], axis=1)
    matrix = np.concatenate([matrix1, matrix2], axis=0)
    initial = np.concatenate([np.eye(2), S], axis=0)

    def expRicatti(t):
        agg_mat = scipy.linalg.expm(matrix*(t-T)) @ initial
        X = agg_mat[:2, :]
        Y = agg_mat[2:, :]
        return Y @ np.linalg.inv(X)

    def u(t, x):
        P = expRicatti(t)
        return -np.linalg.inv(R) @ B.T @ P @ x

        
    def f(t, x, u_func):
        x = np.reshape(x, (2, 1))
        u = u_func(t, x)
        x_dot = A@x + B@u
        return x_dot.flatten()
    
    sim = Simulation(t_0, dt, T, u, f, x_0)

    tvec_ode, xvec_ode = sim.pythonODE(x_0, u)

    tvec_euler, xvec_euler = sim.eulerIntegration(x_0, u)

    uvec = sim.getControlVector(xvec_ode, u)

    plotResults([(tvec_ode, xvec_ode, uvec), (tvec_euler, xvec_euler, uvec)], ['b', 'r--'], ['x1', 'x2', 'u'], "Simulation Results")




if __name__ == '__main__':
    t_0 = 0
    t_f = 10
    dt = 0.1
    x_0 = np.array([1, 2])
    t = np.arange(t_0, t_f, dt)
    t = np.flip(t)

    A = np.array([
        [0, 1],
        [-2, -3]
    ])

    B = np.array([
        [0],
        [1]
    ])

    Q = np.array([
        [1, 0],
        [0, 0]
    ])

    R = np.array([
        [1]
    ])

    S = np.eye(2)

    findP(A, B, Q, R, S, t)

    simulate(A, B, Q, R, S, x_0, t_0, dt, t_f)
    plt.show()





