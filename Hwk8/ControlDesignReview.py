import control as ct
import numpy as np
import sympy as sp
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
            print('System is controllable at eig {}'.format(eig))
        else:
            print('System is not controllable at eig {}'.format(eig))

def controllable_decomposition(A, B):
    control = ct.ctrb(A, B)
    image_space = sc.linalg.orth(control)
    orthogonal_null_space = sc.linalg.null_space(control.T)

    transform = np.concatenate((image_space, orthogonal_null_space), axis=1)

    A_bar = np.linalg.inv(transform) @ A @ transform
    B_bar = np.linalg.inv(transform) @ B

    print('A_bar: ')
    print(*A_bar, sep='\n')
    print('B_bar: ')
    print(*B_bar, sep='\n')
    print('Transform: ')
    print(*transform, sep='\n')

    return A_bar, B_bar, transform

def ctrb(A, B):
    n = A.shape[0]
    gamma = sp.Matrix([((A**i)@B).T for i in range(n)]).T
    return gamma


def problem_12_6():
    print('*************** Problem 12.6 ***************')
    omega = sp.symbols('omega')
    A = sp.Matrix([[0,1,0,0],[3*omega**2,0,0,2*omega],[0,0,0,1],[0,-2*omega,0,1]])
    B = sp.Matrix([[0,0],[1,0],[0,0],[0,1]])

    # a)
    print('a)')
    gamma = ctrb(A, B)
    sp.pprint(gamma.rank())

    # b)
    print('b)')
    print("No radial")
    B_1 = sp.Matrix([[0,0],[0,0],[0,0],[0,1]])
    gamma = ctrb(A, B_1)
    sp.pprint(gamma.rank())

    print('No tangential')
    B_1 = sp.Matrix([[0,0],[1,0],[0,0],[0,0]])
    gamma = ctrb(A, B_1)
    sp.pprint(gamma.rank())
    A = np.array(A.subs(omega, 1)).astype(np.float64)
    B_1 = np.array(B_1).astype(np.float64)
    controlable = controllability(A, B_1)
    analyze_controllability(controlable, A.shape[0])

def problem_12_7():
    print('*************** Problem 12.7 ***************')
    x_1, x_2, x_3, u_1, u_2 = sp.symbols('x_1 x_2 x_3 u_1 u_2')
    x_1_dot = -x_1 + u_1
    x_2_dot = -x_2 + u_2
    x_3_dot = x_2*u_1 - x_1*u_2
    f = sp.Matrix([[x_1_dot], [x_2_dot], [x_3_dot]])
    x = sp.Matrix([[x_1], [x_2], [x_3]])
    u = sp.Matrix([[u_1], [u_2]])

    df_dx = f.jacobian(x)
    df_du = f.jacobian(u)

    # a)
    print('a)')
    A = df_dx.subs({x_1:0, x_2:0, x_3:0, u_1:0, u_2:0})
    B = df_du.subs({x_1:0, x_2:0, x_3:0, u_1:0, u_2:0})

    sp.pprint(A)
    sp.pprint(B)

    print(ctrb(A, B).rank())

    # b)
    print('b)')
    A = df_dx.subs({x_1:1, x_2:1, x_3:1, u_1:1, u_2:1})
    B = df_du.subs({x_1:1, x_2:1, x_3:1, u_1:1, u_2:1})

    sp.pprint(A)
    sp.pprint(B)

    print(ctrb(A, B).rank())

def problem_13_1():
    print('*************** Problem 13.1 ***************')
    A = np.array([[-1, 0], [0, -1]])
    B = np.array([[-1], [1]])
    C = np.eye(2)
    D = np.array([[2], [1]])
    
    s = sp.symbols('s')
    transfer = C@(s*sp.eye(2) - A).inv() @ B + D
    sp.pprint(sp.simplify(transfer))


    controlable = controllability(A, B)
    analyze_controllability(controlable, A.shape[0])

    A_bar, B_bar, transform = controllable_decomposition(A, B)
    C_bar = C @ transform

    print('C_bar: ')
    print(*C_bar, sep='\n')

    A_bar = sp.Matrix(A_bar)
    A_bar[0,1] = 0
    A_bar[1,0] = 0
    B_bar = sp.Matrix(B_bar)
    B_bar[1,0] = 0
    C_bar = sp.Matrix(C_bar)

    s = sp.symbols('s')
    transfer = C_bar@(s*sp.eye(2) - A_bar).inv() @ B_bar + D
    sp.pprint(sp.simplify(transfer))


def problem_A():
    print('*************** Problem A ***************')
    A = np.array([[0, -2], [2, 0]])
    B = np.array([[0], [1]])

    controllable = controllability(A, B)
    analyze_controllability(controllable, A.shape[0])

    K = ct.place(A, B, [-1, -2])

    print("Control law: ", end='')
    print(*K, sep=', ')

    print('Eigenvalues of A - BK: ', end='')
    print(*np.linalg.eigvals(A - B @ K), sep=', ')

def problem_B():
    print('*************** Problem B ***************')
    A = np.array([[1, 0], [0, -1]])
    B = np.array([[0], [1]])

    controlable = controllability(A, B)
    analyze_controllability(controlable, A.shape[0])

    controllable_decomposition(A, B)

def problem_C():
    print('*************** Problem C ***************')
    A = np.array([[-1, 0], [0, 1]])
    B = np.array([[0], [1]])

    controlable = controllability(A, B)
    analyze_controllability(controlable, A.shape[0])
    
    A_bar, B_bar, transform = controllable_decomposition(A, B)

    K = ct.place(A_bar, B_bar, [-1,-2])
    print(K)

def problem_D():
    print('*************** Problem D ***************')
    A = np.array([[-100, 0], [0, -100]])
    B = np.array([[0], [0]])

    controlable = controllability(A, B)
    analyze_controllability(controlable, A.shape[0])



if __name__ == '__main__':
    problem_12_6()
    problem_12_7()
    problem_13_1()
    problem_A()
    problem_B()
    problem_C()
    problem_D()

