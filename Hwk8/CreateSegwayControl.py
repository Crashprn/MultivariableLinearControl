import sympy as sp
import numpy as np
import control as ct
import scipy as sc

def controllability(A, B):
    n = A.shape[0]
    eigs = np.linalg.eigvals(A)
    print('Eigenvalues of A: ', end='')
    print(*eigs, sep=', ')

    controllable = {}
    for eig in eigs:
        Control = np.concatenate((eig * np.eye(n) - A, B), axis=1)
        rank = np.linalg.matrix_rank(Control)
        controllable[eig] = rank

    return controllable


def analyze_controllability(controllable, n):
    for eig, rank in controllable.items():
        if rank == n:
            print('System is controllable at eig {}'.format(eig), end='')
        else:
            print('System is not controllable at eig {}'.format(eig), end='')
        print(' with rank {}'.format(rank))

def controllable_decomposition(A, B):
    control = ct.ctrb(A, B)
    image_space = sc.linalg.orth(control)
    orthogonal_null_space = sc.linalg.null_space(control.T)

    transform = np.concatenate((image_space, orthogonal_null_space), axis=1)

    A_bar = np.linalg.inv(transform) @ A @ transform
    B_bar = np.linalg.inv(transform) @ B

    return A_bar, B_bar, transform

if __name__ == "__main__":
    # Parameters
    m_c, m_s, d, L, R, I_2, I_3, g = sp.symbols('m_c m_s d L R I_2 I_3 g')
    params = {m_c: .503, m_s: 4.315, d:.1, L:.1, R:.073, I_2:.003679, I_3:.02807, g:9.81}

    # States
    x, y, v, v_dot, phi, phi_dot, phi_ddot, psi, omega, omega_dot = sp.symbols('x y v v_dot phi phi_dot phi_ddot psi omega omega_dot')

    # Controls
    u_1, u_2 = sp.symbols('u_1 u_2')

    ## Solve Equations (each fi = 0)
    # Setup Equations
    p = sp.symbols('p') # to substitute in for 1/(2R^2)
    f1 = 3*(m_c+m_s)*v_dot - m_s*d*sp.cos(phi)*phi_ddot + m_s*d*sp.sin(phi)*(phi_dot**2+omega**2) + u_1/R
    f2 = ( (3*L**2 + 1/(2*R**2))*m_c + m_s*d**2*sp.sin(phi)**2 + I_2 )*omega_dot + m_s*d**2*sp.sin(phi)*sp.cos(phi)*omega*phi_dot - L/R*u_2
    f3 = m_s*d*sp.cos(phi)*v_dot + (-m_s*d**2-I_3)*phi_ddot + m_s*d**2*sp.sin(phi)*sp.cos(phi)*phi_dot**2 + m_s*g*d*sp.sin(phi) - u_1


    # Solve Equations
    soln = sp.solve((f1, f2, f3), (v_dot, phi_ddot, omega_dot), dict=True)[0]
    phi_ddot = soln[phi_ddot]
    omega_dot = soln[omega_dot]
    v_dot = soln[v_dot]

    ## Create the generalized linear equations
    # State equations
    print('*****************X*****************')
    X = sp.Matrix([x, y, psi, omega, v, phi, phi_dot])
    U = sp.Matrix([u_1,u_2])
    f = sp.Matrix([[v*sp.cos(psi)], [v*sp.sin(psi)], [omega], [omega_dot], [v_dot], [phi_dot], [phi_ddot]])
    # Calculate the jacobians
    df_dx = f.jacobian(X)
    df_du = f.jacobian(U)
    ## Linearize Equations about x = 0
    # Set the states to zero
    # Evaluate the dynamics at the zero state and control
    equilibrium = {x:0, y:0, psi:0, omega:0, v:0, phi:0, phi_dot:0, u_1: 0, u_2:0}
    sp.pprint(f.subs(equilibrium))
    
    # Create the state matrices (A,B) from df_dx and df_du
    A = df_dx.subs(equilibrium)
    B = df_du.subs(equilibrium)

    # Substitute in the parameters
    A_1 = A.subs(params)
    B_1 = B.subs(params)

    # Print linearized A and B
    sp.pprint(A_1)
    sp.pprint(B_1)

    # Calculate controllability
    A_1 = np.array(A_1).astype(np.float64)
    B_1 = np.array(B_1).astype(np.float64)

    controllable = controllability(A_1, B_1)
    analyze_controllability(controllable, A_1.shape[0])

    # Find decomposition
    A_bar, B_bar, transform = controllable_decomposition(A_1, B_1)
    A_bar = sp.Matrix(A_bar)
    B_bar = sp.Matrix(B_bar)

    sp.pprint(A_bar)
    sp.pprint(B_bar)



    print('*****************Z*****************')
    z = sp.Matrix([omega, v, phi, phi_dot])
    f = sp.Matrix([[omega_dot], [v_dot], [phi_dot], [phi_ddot]])
    df_dz = f.jacobian(z)
    df_du = f.jacobian(U)

    equillibrium = {omega:0, v:0, phi:0, phi_dot:0, u_1:0, u_2:0}

    A = df_dz.subs(equilibrium)
    B = df_du.subs(equilibrium)

    A_1 = A.subs(params)
    B_1 = B.subs(params)

    sp.pprint(A_1)
    sp.pprint(B_1)

    A_1 = np.array(A_1).astype(np.float64)
    B_1 = np.array(B_1).astype(np.float64)

    controllable = controllability(A_1, B_1)
    analyze_controllability(controllable, A_1.shape[0])

    K = ct.place(A_1, B_1, [-1, -2, -3, -4])
    print('K = ')
    print(K)

    print('Eigenvalues of A - BK: ', end='')
    print(*np.linalg.eigvals(A_1 - B_1 @ K), sep=', ')

    
