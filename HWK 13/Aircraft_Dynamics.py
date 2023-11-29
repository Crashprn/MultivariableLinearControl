import numpy as np
import control as ct
import scipy
from Simulation import Simulation, plotResults
import matplotlib.pyplot as plt

def get_system_matrices():
    """Returns the system matrices for the aircraft dynamics model.

    Returns:
        A (numpy.ndarray): The A matrix.
        B (numpy.ndarray): The B matrix.
    """
    A = np.array([
        [-0.038, 18.984, 0, -32.174],
        [-0.001, -0.632, 1, 0],
        [0, -0.759, -0.518, 0],
        [0, 0, 1, 0]
    ])

    B = np.array([
        [10.1, 0],
        [0, -0.0086],
        [0.025, -0.011],
        [0, 0]
    ])
    
    return A, B

def control_decomposition(A, B, Q, R):
    gamma = ct.ctrb(A, B)
    print(f'Rank of gamma: {np.linalg.matrix_rank(gamma)}')

    V = scipy.linalg.orth(gamma)
    W = scipy.linalg.null_space(gamma.T)

    T = np.concatenate([V, W], axis=1)

    A_bar = np.linalg.inv(T) @ A @ T
    B_bar = np.linalg.inv(T) @ B

    A_11 = A_bar[:V.shape[1], :V.shape[1]]
    B_1 = B_bar[:V.shape[1], :]

    k = ct.lqr(A_11, B_1, Q, R)[0]

    k = np.concatenate([k, np.zeros((k.shape[0], W.shape[1]))], axis=1)

    k_full = k @ np.linalg.inv(T)

    print(f'k: \n{k_full}')

    eigs = np.linalg.eigvals(A - B @ k_full)
    print(f'Eigenvalues of A - B @ k:')
    print(*eigs)

    return k_full

def simulate(A, B, K, x_0, t_0, dt, t_f):
    """Simulates the aircraft dynamics model.

    Args:
        A (numpy.ndarray): The A matrix.
        B (numpy.ndarray): The B matrix.
        Q (numpy.ndarray): The Q matrix.
        R (numpy.ndarray): The R matrix.
        S (numpy.ndarray): The S matrix.
        M (numpy.ndarray): The M matrix.
        x_0 (numpy.ndarray): The initial state vector.
        t_0 (float): The initial time.
        dt (float): The time step.
        t_f (float): The final time.
    """
    def dynamics(t, x, u_func):
        x = x.reshape((x.shape[0], 1))
        u = u_func(t, x)
        return (A @ x + B @ u).flatten()
    
    def u_func(t, x):
        return -K @ x

    sim = Simulation(t_0, dt, t_f, u_func, dynamics, x_0, u_dim=2)
    t_vec, x_vec = sim.pythonODE(x_0, u_func)
    u_vec = sim.getControlVector(x_vec, u_func)

    plotResults([(t_vec, x_vec, u_vec)], ['b'], ['V', 'alpha', 'q', 'theta'], 'Aircraft Simulation', legend=False)
    plt.show()

if __name__ == '__main__':
    A, B = get_system_matrices()

    Q = np.diag([1/10**2, 1/.1**2, 1/.1**2, 1/.05**2])
    R = np.diag([1/(np.pi/2)**2, 1/(np.pi/2)**2])
    k = control_decomposition(A, B, Q, R)
    x_0 = np.array([10, .1, .1, 0])
    t_0 = 0
    dt = 0.01
    t_f = 60

    simulate(A, B, k, x_0, t_0, dt, t_f)
