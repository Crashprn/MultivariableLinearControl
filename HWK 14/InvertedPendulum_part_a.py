import numpy as np
import matplotlib.pyplot as plt
import control as ct
from Simulation import Simulation
import sympy as sp

class params:
    def __init__(self):
        self.g = 0
        self.l = 0
        self.m = 0
        self.b = 0
        self.K = 0
        self.C = 0
        self.xd = 0
        self.u_ff = 0

def InvertedPendulum_part_a():
    # Define parameters
    P = params()
    P.g = 9.8 # Gravity constant
    P.l = 0.25 # Pendulum length
    P.m = 1/9.8 # Pendulum mass
    P.b = 1.0 # Friction coefficient
    
    Q = np.diag([1/.15**2, 1/.5**2])
    R = np.array([[1]])
    # Setup for convergences to pi/4ths
    P = controlParamters_PiFourths_a(P, Q, R) # Create control
    sp.pprint(sp.Matrix(P.K))
    sp.pprint(sp.Matrix(P.u_ff))
    u = lambda t, x: control_PiFourths_a(t, x, P) # Set the control handle
    x0 = np.array([np.pi/4 - 0.15, 0]) # Initial state  
    
    # Simulate the state forward in time the state
    dt = 0.01
    t_0 = 0
    t_f = 10

    sim = Simulation(t_0, dt, t_f, 1)

    sim.pythonODE(lambda t, x: f_ideal(t, x, u(t,x), P), x0)
    sim.getControlVector(u)
    sim.save_last_sim()

    sim.pythonODE(lambda t, x: f_true_a(t, x, u(t,x), P), x0)
    sim.getControlVector(u)
    sim.save_last_sim()


    # Plot the results
    sim.plotResults([0, 1, 2], ['r', 'b'], ['theta', 'theta_dot', 'u'],
                    'Ideal Simulation', all_sims=True, sim_names=['Ideal', 'True'])
    plt.show()

def InvertedPendulum_part_b():
    # Define parameters
    P = params()
    P.g = 9.8 # Gravity constant
    P.l = 0.25 # Pendulum length
    P.m = 1/9.8 # Pendulum mass
    P.b = 1.0 # Friction coefficient
    P.C = np.array([[1, 0, 0]])
    
    Q = np.diag([1/.15**2, 1/.5**2, 1/.2**2])
    R = np.array([[1]])
    # Setup for convergences to pi/4ths
    P = controlParamters_PiFourths_b(P, Q, R) # Create control
    sp.pprint(sp.Matrix(P.K))
    sp.pprint(sp.Matrix(P.u_ff))
    u = lambda t, x: control_PiFourths_b(t, x, P) # Set the control handle
    x0 = np.array([np.pi/4 - 0.15, 0, 0]) # Initial state  
    
    # Simulate the state forward in time the state
    dt = 0.01
    t_0 = 0
    t_f = 10

    sim = Simulation(t_0, dt, t_f, 1)

    sim.pythonODE(lambda t, x: f_true_b(t, x, u(t,x), P), x0)
    sim.getControlVector(u)
    sim.save_last_sim()

    # Plot the results
    sim.plotResults([0, 1, 3], ['b'], ['theta', 'theta_dot', 'u'], 'Integral Simulation')
    plt.show()
    
def f_ideal(t,x, u, P):
    # Define parameters
    g = P.g
    l = P.l
    m = P.m
    b = P.b
    
    # Extract state
    theta = x[0]
    thetadot = x[1]
    
    # Saturate the input
    u = max(u,-1)
    u = min(u, 1)
    
    # Calculate dynamics
    xdot = np.zeros((2,1))
    xdot[0] = thetadot
    xdot[1] = g/l*np.sin(theta) - b/(m*l**2)*thetadot + 1/(m*l**2)*u

    return xdot.flatten().squeeze()

def f_true_a(t, x, u, P):
    # Define parameters
    g = P.g
    l = P.l + 0.1
    m = P.m + 0.2
    b = P.b - 0.1
    
    # Extract state
    theta = x[0]
    thetadot = x[1]
    
    # Saturate the input
    u = max(u,-1)
    u = min(u, 1)
    
    # Calculate dynamics
    xdot = np.zeros((2,1))
    xdot[0] = thetadot
    xdot[1] = g/l*np.sin(theta) - b/(m*l**2)*thetadot + 1/(m*l**2)*u

    return xdot.flatten().squeeze()

def f_true_b(t, x, u, P):
    # Define parameters
    g = P.g
    l = P.l + 0.1
    m = P.m + 0.2
    b = P.b - 0.1
    
    # Extract state
    theta = x[0]
    thetadot = x[1]
    
    # Saturate the input
    u = max(u,-1)
    u = min(u, 1)
    
    # Calculate dynamics
    xdot = np.zeros((3,1))
    xdot[0] = thetadot
    xdot[1] = g/l*np.sin(theta) - b/(m*l**2)*thetadot + 1/(m*l**2)*u
    xdot[2] = P.C@(x.reshape(-1,1) - P.xd) 

    return xdot.flatten().squeeze()

def controlParamters_PiFourths_a(P, Q, R):
    # Define the linearized dynamics
    A = np.array([
        [0, 1],
        [np.sqrt(2)/2*P.g/P.l, -P.b/(P.m*P.l**2)]
    ])

    B = np.array([
        [0],
        [1/(P.m*P.l**2)]
    ])
 
    # Design the control    
    P.K = ct.lqr(A, B, Q, R)[0]
    
    # Define the desired value
    P.xd = np.array([
        [np.pi/4],
        [0]
    ])
    
    # Create feedforward term
    u_ff = sp.symbols('u_ff')
    eq = B*u_ff + A@P.xd
    u_ff_sol = sp.solve(eq[1], u_ff, dict=True)[0]
    P.u_ff = np.array([u_ff_sol[u_ff]]).astype(np.float64)

    return P

def controlParamters_PiFourths_b(P, Q, R):
    # Define the linearized dynamics
    A = np.array([
        [0, 1],
        [np.sqrt(2)/2*P.g/P.l, -P.b/(P.m*P.l**2)]
    ])

    A_hat = np.concatenate([
        np.concatenate((A, np.zeros((2,1))), axis=1),
        P.C]
        , axis=0)

    B = np.array([
        [0],
        [1/(P.m*P.l**2)]
    ])

    B_hat = np.concatenate([
        B,
        [[0]]
    ], axis=0)
 
    # Design the control    
    P.K = ct.lqr(A_hat, B_hat, Q, R)[0]
    
    # Define the desired value
    P.xd = np.array([
        [np.pi/4],
        [0],
        [0]
    ])
    
    # Create feedforward term
    u_ff = sp.symbols('u_ff')
    eq = B*u_ff + A@P.xd[:2, :]
    u_ff_sol = sp.solve(eq[1], u_ff, dict=True)[0]
    P.u_ff = np.array([u_ff_sol[u_ff]]).astype(np.float64)

    return P

def control_verticle(t, x, P):
    u = -P.K@x
    
    # Saturate the input
    u = max(u,-1)
    u = min(u, 1)

    return u


def control_PiFourths_a(t, x, P):
    # Create difference in theta
    dtheta = x[0] - np.pi/4
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta)) # adjust difference to be between -pi and pi
    
    # Calcuate control
    dx = np.array([
        [dtheta],
        [x[1]]
    ])

    u = P.u_ff - P.K@dx
    
    # Saturate the input
    u = np.max(u, initial=-1)
    u = np.min(u, initial=1)

    return np.array([u])

def control_PiFourths_b(t, x, P):
    # Create difference in theta
    dtheta = x[0] - np.pi/4
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta)) # adjust difference to be between -pi and pi
    
    # Calcuate control
    dx = np.array([
        [dtheta],
        [x[1]],
        [x[2]]
    ])

    u = P.u_ff - P.K@dx
    
    # Saturate the input
    u = np.max(u, initial=-1)
    u = np.min(u, initial=1)

    return np.array([u])

if __name__ == "__main__":
    InvertedPendulum_part_a()
    InvertedPendulum_part_b()