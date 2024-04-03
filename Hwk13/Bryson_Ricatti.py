import numpy as np
from scipy.integrate import odeint
import scipy
import typing as t
import matplotlib.pyplot as plt
import control as ct
import sympy as sp
from Simulation import Simulation, plotResults

def controllable_decomposition(A, B):
    gamma = ct.ctrb(A, B)
    print(f'Rank of Gamma: {np.linalg.matrix_rank(gamma)}')
    print(f'Gamma: \n{gamma}')

    V = scipy.linalg.orth(gamma)

    W = scipy.linalg.null_space(gamma.T)

    T = np.concatenate([V, W], axis=1)

    A_bar = np.linalg.inv(T) @ A @ T
    B_bar = np.linalg.inv(T) @ B

    print(f'A_bar: \n{A_bar}')
    print(f'B_bar: \n{B_bar}')


def bryson_ricatti():
    A = np.array([
        [3, 6, 4],
        [9, 6, 10],
        [-7, -7, -9]
    ])

    B = np.array([
        [-2/3, 1/3],
        [1/3, -2/3],
        [1/3, 1/3]
    ])

    t_0 = 0
    t_f = 1
    dt = 0.01

    controllable_decomposition(A, B)

    Q = np.diag([1, 1/100**2, 1/100**2])
    R = np.diag([1/5**2, 1/10**2])
    S = np.diag([1/10**2, 1, 1/2**2])

    matrix1 = np.concatenate([A, -B@np.linalg.inv(R)@B.T], axis=1)
    matrix2 = np.concatenate([-Q, -A.T], axis=1)
    M = np.concatenate([matrix1, matrix2], axis=0)

    t = np.arange(t_0, t_f, dt)
    t = np.flip(t)
    x_0 = np.array([5, 3, 2])

    findP(A, B, Q, R, S, M, t)
    simulate(A, B, Q, R, S, M, x_0, t_0, dt, t_f, 'Bryson Ricatti')

    S_new = np.array([
        [10, 0, 0],
        [0, 1, 0],
        [0, 0, .25]
    ])
    simulate(A, B, Q, R, S_new, M, x_0, t_0, dt, t_f, 'Bryson Ricatti with new S')

    plt.show()


def findP(A, B, Q, R, S, M, t):
    n = A.shape[0]
    def f(x, t):
        P = np.reshape(x, (n, n))
        P_dot = -A.T @ P - P @ A - Q + P @ B @ np.linalg.inv(R) @ B.T @ P 

        return P_dot.flatten()
    
    P_regular = odeint(f, S.flatten(), t)

    def f(x, t):
        x = np.reshape(x, (n*2, n))

        x_dot = M @ x
        
        return x_dot.flatten()

    x_0 = np.concatenate([np.eye(n), S]).flatten()
    x_numeric = odeint(f, x_0, t).reshape(-1, n*2, n)
    X = x_numeric[:, :n, :]
    Y = x_numeric[:, n:, :]
    P_numeric = np.zeros((len(t), n, n))
    for i, (x,y) in enumerate(zip(X, Y)):
        P_numeric[i] = y @ np.linalg.inv(x)
    P_numeric = P_numeric.reshape(-1, n*n)

    plot_result([P_regular, P_numeric], t)
    plot_diff(P_regular - P_numeric, t)

def simulate(A, B, Q, R, S, M, x_0, t_0, dt, T, title):
    initial = np.concatenate([np.eye(3), S], axis=0)

    def expRicatti(t):
        agg_mat = scipy.linalg.expm(M*(t-T)) @ initial
        X = agg_mat[:3, :]
        Y = agg_mat[3:, :]
        return Y @ np.linalg.inv(X)

    def u(t, x):
        P = expRicatti(t)
        return -np.linalg.inv(R) @ B.T @ P @ x

        
    def f(t, x, u_func):
        x = np.reshape(x, (3, 1))
        u = u_func(t, x)
        x_dot = A@x + B@u
        return x_dot.flatten()
    
    sim = Simulation(t_0, dt, T, u, f, x_0, u_dim=2)

    tvec_ode, xvec_ode = sim.pythonODE(x_0, u)

    tvec_euler, xvec_euler = sim.eulerIntegration(x_0, u)

    uvec = sim.getControlVector(xvec_ode, u)

    plotResults([(tvec_ode, xvec_ode, uvec), (tvec_euler, xvec_euler, uvec)], ['b', 'r--'], ['x1', 'x2', 'x3'], title)

def plot_result(data, t):
    labels = ['P_11', 'P_12', 'P_13', 'P_12', 'P_22', 'P_23', 'P_13', 'P_23', 'P_33']
    generated_label = ['Regular', 'Numeric']
    style = ['b', 'r--']
    fig, ax = plt.subplots(len(labels), 1, figsize=(10, 10))
    for i, dat in enumerate(data):
        for j in range(dat.shape[1]):
            ax[j].plot(t, dat[:, j], style[i], label=generated_label[i])
            ax[j].set_xlabel('Time (s)')
            ax[j].set_title(labels[j])
            ax[j].legend()


def plot_diff(data, t):
    labels = ['P_11 Error', 'P_12 Error', 'P_13_Error', 'P_12 Error', 'P_22 Error', 'P_23 Error', 'P_13 Error', 'P_23 Error', 'P_33 Error']
    fig, ax = plt.subplots(len(labels), 1, figsize=(10, 10))

    for i in range(data.shape[1]):
        ax[i].plot(t, data[:, i])
        ax[i].set_xlabel('Time (s)')
        ax[i].set_title(labels[i])


if __name__ == '__main__':
    bryson_ricatti()
