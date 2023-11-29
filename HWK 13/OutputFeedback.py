import numpy as np
import sympy as sp
import control as ct
import scipy


def find_uff(A, B, x_d):
    u_ff_1, u_ff_2, u_ff_3 = sp.symbols('u_ff_1 u_ff_2 u_ff_3')
    u_ff = sp.Matrix([[u_ff_1], [u_ff_2], [u_ff_3]])

    solution = sp.solve(B @ u_ff + A @ x_d, u_ff, dict=True)[0]

    sp.pprint(solution)

    return np.array(u_ff.subs(solution)).astype(np.float64)

def control_decomposition(A, B):
    gamma = ct.ctrb(A, B)
    print(f'Rank of gamma: {np.linalg.matrix_rank(gamma)}')

    V = scipy.linalg.orth(gamma)
    W = scipy.linalg.null_space(gamma.T)

    T = np.concatenate([V, W], axis=1)

    A_bar = np.linalg.inv(T) @ A @ T
    B_bar = np.linalg.inv(T) @ B

    print(f'A_bar:')
    sp.pprint(sp.Matrix(A_bar), full_prec=False)
    print(f'B_bar:')
    sp.pprint(sp.Matrix(B_bar), full_prec=False)

    A_11 = A_bar[:V.shape[1], :V.shape[1]]
    B_1 = B_bar[:V.shape[1], :]

    Q = np.diag([1, 2, 3, 4])
    R = np.diag([1, 2, 3])

    k = ct.lqr(A_11, B_1, Q, R)[0]

    k = np.concatenate([k, np.zeros((3, 1))], axis=1)

    k_full = k @ np.linalg.inv(T)

    print(f'k: \n{k_full}')

    eigs = np.linalg.eigvals(A - B @ k_full)
    print(f'Eigenvalues of A - B @ k:')
    print(*eigs)

    return k_full

def observable_decomposition(A, C):
    omega = ct.obsv(A, C)
    print(f'Rank of omega: {np.linalg.matrix_rank(omega)}')

    V = scipy.linalg.orth(omega.T)
    W = scipy.linalg.null_space(omega)

    T = np.concatenate([V, W], axis=1)

    A_bar = np.linalg.inv(T) @ A @ T
    C_bar = C @ T

    print(f'A_bar:')
    sp.pprint(sp.Matrix(A_bar))
    print(f'C_bar:')
    sp.pprint(sp.Matrix(C_bar))

    A_11 = A_bar[:V.shape[1], :V.shape[1]]
    C_1 = C_bar[:, :V.shape[1]]

    L = ct.lqr(A_11.T, C_1.T, np.diag([1, 2, 3, 4]), np.diag([1, 2, 3]))[0]

    L_full = np.concatenate([L, np.zeros((3, 1))], axis=1).T

    L_full = np.linalg.inv(T).T @ L_full

    print(f'L_full: \n{L_full}')

    eigs = np.linalg.eigvals(A - L_full @ C)

    print(f'Eigenvalues of A - L @ C:')
    print(*eigs)

    return L_full



def solve():
    A = np.array([
        [3, 0, -4, 0 , 0],
        [0, 2.2, 0.8, 0, 3.6],
        [0, 0, -1, 0, 0],
        [1, 0, 4, 4, 0],
        [0, -1.4, -0.6, 0, -3.2]
    ])

    B = np.array([
        [0, 2, 0],
        [0.8, .2, .6],
        [1, 1, 0],
        [0, -1, 0],
        [-.6, -.4, -.2]
    ])

    C = np.array([
        [1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 2, 0, 0, 1]
    ])

    x_d = np.array([
        [-10/3],
        [3],
        [-3],
        [43/12],
        [0]
    ])

    u_ff = find_uff(A, B, x_d)

    k = control_decomposition(A, B)

    L = observable_decomposition(A, C)











if __name__ == "__main__":
    solve()