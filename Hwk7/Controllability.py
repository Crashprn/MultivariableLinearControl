import sympy as sp
import control as ct
import numpy as np

def controllability(A, B):
    n = A.shape[0]
    eigs = A.eigenvals().keys()
    print('Eigenvalues of A: ', end='')
    print(*eigs, sep=', ')

    for eig in eigs:
        Control = sp.Matrix([(eig * sp.eye(n) - A).T, B.T]).T
        print('Rank of Augmented Eig {}: {}'.format(eig, Control.rank()))

def solve_k(A, B, K, syms, poly_coeff):
    n = A.shape[0]
    s = sp.symbols('s')
    A = s*sp.eye(n) - (A - B@K)
    sp.pprint(A)
    d = A.det() 
    coeffs = sp.Poly(d, s).coeffs()
    eqs = [sp.Eq(sol,c) for sol, c in zip(poly_coeff, coeffs[1:])]
    k_sol = sp.solve(eqs, syms, dict=True)[0]
    print(k_sol)
    return k_sol


def problem_2_3():
    # Problem 2.3
    print('**********************Problem 2.3**********************')
    x_1 = sp.symbols('x_1')
    A = sp.Matrix([[0, 1], [4 * 9.8 * sp.cos(x_1), -16*9.8]])
    B = sp.Matrix([[0], [16*9.8]])

    # a)
    print('a)')
    A_1 = A.subs({'x_1': 0})
    controllability(A_1, B)

    # b)
    print('b)')
    A_2 = A.subs({'x_1': sp.pi})
    controllability(A_2, B)

    # c)
    print('c)')
    A_3 = A.subs({'x_1': sp.pi/4})
    controllability(A_3, B)

def problem_2_4():
    # Problem 2.4
    print('**********************Problem 2.4**********************')
    A = sp.Matrix([[0, 1], [-1, 0]])
    B = sp.Matrix([[0], [1]])
    controllability(A, B)

def problem_2_6():
    # Problem 2.6
    print('**********************Problem 2.6**********************')
    A = sp.Matrix([[0, 1], [1, -2]])
    B = sp.Matrix([[0], [1]])
    controllability(A, B)

def problem_2_7():
    # Problem 2.7
    print('**********************Problem 2.7**********************')
    x_1, x_2, u_1, u_2 = sp.symbols('x_1, x_2, u_1, u_2')
    A = sp.Matrix([[0, u_2, 0],[-u_2, 0, 0],[0, 0, 0]])
    B = sp.Matrix([[1, x_2], [-x_1, 0], [0, 1]])

    # b)
    print('b')
    vals = {'x_1': 0, 'x_2': 0, 'u_1': 0, 'u_2': 0}
    A_1 = A.subs(vals)
    B_1 = B.subs(vals)
    controllability(A_1, B_1)

    # c)
    print('c')
    vals = {'x_1': 0, 'x_2': -1, 'u_1': 1, 'u_2': 1}
    A_2 = A.subs(vals)
    B_2 = B.subs(vals)
    controllability(A_2, B_2)

def P2_S1():
    # Problem 2 -Section 1
    print('**********************Problem 2 - Section 1**********************')
    syms = sp.symbols('k_1, k_2')
    A = sp.Matrix([[1, 2], [3, 4]])
    B = sp.Matrix([0, 1])
    K = sp.Matrix([[syms[0], syms[1]]])

    controllability(A, B)

    k_s = solve_k(A, B, K, syms, [3, 2])
    K = K.subs(k_s)
    eigs = (A - B@K).eigenvals()
    print("Eigenvalues of A - BK: ", end='')
    print(*eigs, sep=', ')

def P2_S2():
    # Problem 2 -Section 2
    print('**********************Problem 2 - Section 2**********************')
    syms = sp.symbols('k_1, k_2')
    A = sp.Matrix([[-1, -2], [6, 7]])
    B = sp.Matrix([-.5, 1])
    K = sp.Matrix([[syms[0], syms[1]]])
    
    controllability(A, B)

    k_s = solve_k(A, B, K, syms, [3, 2])
    K = K.subs(k_s)
    eigs = (A - B@K).eigenvals()
    print("Eigenvalues of A - BK: ", end='')
    print(*eigs, sep=', ')


def P2_S3():
    # Problem 2 -Section 3
    print('**********************Problem 2 - Section 3**********************')
    A = sp.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = sp.Matrix([[1, 0], [0, 1], [1, 1]])
   
    controllability(A, B)
    
    A = np.array(A.tolist()).astype(np.float64)
    B = np.array(B.tolist()).astype(np.float64)
    K = ct.place(A, B, [-1,-2,-3])
    print(K)

    print(np.linalg.eig(A-B@K)[0])





if __name__ == '__main__':
    problem_2_3()
    print()
    problem_2_4()
    print()
    problem_2_6()
    print()
    problem_2_7()
    P2_S1()
    P2_S2()
    P2_S3()