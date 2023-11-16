import numpy as np
import control as ct
import typing as t
import sympy as sp


class Segway:

    def __init__(self, v_d: float, omega_d: float, A: np.ndarray, B: np.ndarray, dynamics, feedback_poles=np.array([-1.1, -1.2, -1.3, -1.4])):
        self.A = A
        self.B = B
        self.C = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])

        self.d_x1 = dynamics[0]
        self.d_x2 = dynamics[1]
        self.d_omega_dot = dynamics[2]
        self.d_v_dot = dynamics[3]
        self.d_phi_ddot = dynamics[4]

        self.n = 7
        self.ind_x1 = 0
        self.ind_x2 = 1
        self.ind_psi = 2
        self.ind_omega = 3
        self.ind_v = 4
        self.ind_phi = 5
        self.ind_phi_dot = 6

        self.z_ind = [self.ind_omega, self.ind_v, self.ind_phi, self.ind_phi_dot]
        self.z_ind_omega = 0
        self.z_ind_v = 1
        self.z_ind_phi = 2
        self.z_ind_phi_dot = 3

        self.v_d = v_d
        self.omega_d = omega_d
        self.x_d = np.array([self.omega_d, self.v_d, 0, 0]).T
        self.K = self.create_feedback_controller(feedback_poles)
        self.u_ff = self.create_feedforward_controller()
        self.L = self.create_observer(10*feedback_poles)
    
    def create_feedback_controller(self, feedback_poles) -> np.ndarray:
        return np.array(ct.place(self.A, self.B, feedback_poles))
    
    def create_observer(self, poles) -> np.ndarray:
        return np.array(ct.place(self.A.T, self.C.T, poles)).T
    
    def create_feedforward_controller(self) -> np.ndarray:
        u_ff_1 , u_ff_2 = sp.symbols('u_ff_1 u_ff_2')
        u_ff = sp.Matrix([[u_ff_1], [u_ff_2]])

        u_ff_sol = sp.solve(self.B@u_ff - (self.A@self.x_d).reshape(4,1), (u_ff_1, u_ff_2), dict=True)[0]

        return np.array([u_ff_sol[u_ff_1], u_ff_sol[u_ff_2]]).T


    def get_feedback_control(self, t, x) -> t.Tuple[float, float]:
        z = x[self.z_ind].T

        u_total = -self.K@(z-self.x_d) + self.u_ff

        return u_total.T

    def observer_dynamics(self, x_hat, u, y):
        L = self.L@(y - self.C@x_hat)
        A = self.A@x_hat
        B = self.B@u
        x_hat_dot = A + B.squeeze() + L
        return x_hat_dot.T

    def x_dot(self, t, x, u_1, u_2) -> np.ndarray: 
        psi = x[self.ind_psi]
        omega = x[self.ind_omega]
        v = x[self.ind_v]
        phi = x[self.ind_phi]
        phi_dot = x[self.ind_phi_dot]

        xdot = np.zeros(self.n)

        xdot[self.ind_x1] = self.d_x1(v, psi)
        xdot[self.ind_x2] = self.d_x2(v, psi)
        xdot[self.ind_psi] = omega
        xdot[self.ind_omega] = self.d_omega_dot(omega, phi, phi_dot, u_2)
        xdot[self.ind_v] = self.d_v_dot(omega, phi, phi_dot, u_1)
        xdot[self.ind_phi] = phi_dot
        xdot[self.ind_phi_dot] = self.d_phi_ddot(omega, phi, phi_dot, u_1)

        return xdot

    def x_dot_w_observer(self, t, x):
        # Extract States
        x_state = x[:self.n]
        x_hat = x[self.n:]

        # Get Control
        u_1, u_2 = self.get_feedback_control(t, x_hat)

        # Calculate state dynamics
        x_dot_state = self.x_dot(t, x_state, u_1, u_2)

        # Calculate measurement
        y = self.C @ x_state[self.z_ind]

        # calculate z_hat dynamics
        x_hat_dot = self.observer_dynamics(x_hat[self.z_ind].T, np.array([[u_1],[u_2]]).astype(np.float64), y)

        return np.concatenate((x_dot_state, np.zeros(3), x_hat_dot))






    


