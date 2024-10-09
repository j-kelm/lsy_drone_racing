
import numpy as np
import casadi as cs

from lsy_drone_racing.sim.symbolic import SymbolicModel
from lsy_drone_racing.sim.symbolic import csRotXYZ

# thrust2weight=2.25
DRONE_PROP = {
    'L': 0.0397,
    'g': 9.8,
}

INFO = {
    'nominal_physical_parameters': {
        'quadrotor_mass': 0.03454,
        'quadrotor_ixx_inertia': 1.4e-05,
        'quadrotor_iyy_inertia': 1.4e-05,
        'quadrotor_izz_inertia': 2.17e-05,
    },
    'x_reference': np.array([ 0. ,  0. , -2. ,  0. ,  0.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , 0. ]),
    'u_reference': np.array([0.084623, 0.084623, 0.084623, 0.084623]),
    'ctrl_timestep': 0.03333333333333333,
    'ctrl_freq': 30,
    'episode_len_sec': 33,
    'quadrotor_kf': 3.16e-10,
    'quadrotor_km': 7.94e-12,
    'gate_dimensions': {
        'tall': {'shape': 'square', 'height': 1.0, 'edge': 0.45},
        'low': {'shape': 'square', 'height': 0.525, 'edge': 0.45}
    },
    'obstacle_dimensions': {'shape': 'cylinder', 'height': 1.05, 'radius': 0.05},
    'nominal_gates_pos_and_type': [
        [-0.5, -0.5, 0, 0, 0, 3.14, 0]
    ],
    'nominal_obstacles_pos': [],
}


class Model:
    def __init__(self, info):
        self.info = info if info is not None else INFO

        self.dt = self.info['ctrl_timestep']


        physical_params = self.info['nominal_physical_parameters']
        self.m = physical_params['quadrotor_mass']
        self.Ixx = physical_params['quadrotor_ixx_inertia']
        self.Iyy = physical_params['quadrotor_iyy_inertia']
        self.Izz = physical_params['quadrotor_izz_inertia']

        self.KF = self.info['quadrotor_kf']
        self.KM = self.info['quadrotor_km']


        self.setup_symbolics()

        self.input_bounds = None
        self.state_bounds = None
        self.input_constraints = []
        self.state_constraints = []
        self.input_constraints_soft = []
        self.state_constraints_soft = []

        self.STATE_LABELS = ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot',
                             'phi', 'theta', 'psi', 'p', 'q', 'r']
        self.STATE_UNITS = ['m', 'm', 'm', 'm/s','m/s', 'm/s',
                            'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s']
        self.INPUT_LABELS = ['T1',  'T2', 'T3', 'T4']
        self.INPUT_UNITS = ['N', 'N', 'N', 'N']

    def setup_symbolics(self):
        nx, nu = 12, 4

        # set drone properties
        g, length = DRONE_PROP['g'], DRONE_PROP['L']

        J = cs.blockcat([[self.Ixx, 0.0, 0.0],
                         [0.0, self.Iyy, 0.0],
                         [0.0, 0.0, self.Izz]])
        Jinv = cs.blockcat([[1.0 / self.Ixx, 0.0, 0.0],
                            [0.0, 1.0 / self.Iyy, 0.0],
                            [0.0, 0.0, 1.0 / self.Izz]])

        u_eq = self.m * g
        gamma = self.KM / self.KF

        x = cs.MX.sym('x')
        y = cs.MX.sym('y')
        z = cs.MX.sym('z')
        x_dot = cs.MX.sym('x_dot')
        y_dot = cs.MX.sym('y_dot')
        z_dot = cs.MX.sym('z_dot')
        phi = cs.MX.sym('phi')  # Roll
        theta = cs.MX.sym('theta')  # Pitch
        psi = cs.MX.sym('psi')  # Yaw
        p_body = cs.MX.sym('p')  # Body frame roll rate
        q_body = cs.MX.sym('q')  # body frame pitch rate
        r_body = cs.MX.sym('r')  # body frame yaw rate
        # PyBullet Euler angles use the SDFormat for rotation matrices.
        Rob = csRotXYZ(phi, theta, psi)  # rotation matrix transforming a vector in the body frame to the world frame.

        # Define state variables.
        X = cs.vertcat(x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, p_body, q_body, r_body)

        # Define inputs.
        f1 = cs.MX.sym('f1')
        f2 = cs.MX.sym('f2')
        f3 = cs.MX.sym('f3')
        f4 = cs.MX.sym('f4')
        U = cs.vertcat(f1, f2, f3, f4)

        # From Ch. 2 of Luis, Carlos, and Jérôme Le Ny. 'Design of a trajectory tracking controller for a
        # nanoquadcopter.' arXiv preprint arXiv:1608.05786 (2016).

        # Defining the dynamics function.
        # We are using the velocity of the base wrt to the world frame expressed in the world frame.
        # Note that the reference expresses this in the body frame.
        oVdot_cg_o = Rob @ cs.vertcat(0, 0, f1 + f2 + f3 + f4) / self.m - cs.vertcat(0, 0, g)
        pos_ddot = oVdot_cg_o
        pos_dot = cs.vertcat(x_dot, y_dot, z_dot)
        Mb = cs.vertcat(length / cs.sqrt(2.0) * (f1 + f2 - f3 - f4),
                        length / cs.sqrt(2.0) * (-f1 + f2 + f3 - f4),
                        gamma * (f1 - f2 + f3 - f4))
        rate_dot = Jinv @ (Mb - (cs.skew(cs.vertcat(p_body, q_body, r_body)) @ J @ cs.vertcat(p_body, q_body, r_body)))
        ang_dot = cs.blockcat([[1, cs.sin(phi) * cs.tan(theta), cs.cos(phi) * cs.tan(theta)],
                                [0, cs.cos(phi), -cs.sin(phi)],
                                [0, cs.sin(phi) / cs.cos(theta), cs.cos(phi) / cs.cos(theta)]]) @ cs.vertcat(p_body, q_body, r_body)
        X_dot = cs.vertcat(pos_dot, pos_ddot, ang_dot, rate_dot)

        Y = cs.vertcat(x, y, z, x_dot, y_dot, z_dot, pos_ddot, phi, theta, psi, ang_dot, p_body, q_body, r_body)
        # Set the equilibrium values for linearizations.

        self.X_EQ = np.zeros(nx)
        self.U_EQ = np.ones(nu) * u_eq / nu

        # Define cost (quadratic form).
        q = cs.MX.sym('Q', nx)
        r = cs.MX.sym('R', nu)
        Xr = cs.MX.sym('Xr', nx, 1)
        Ur = cs.MX.sym('Ur', nu, 1)
        cost_func = 0.5 * (X - Xr).T @ cs.diag(q) @ (X - Xr) + 0.5 * (U - Ur).T @ cs.diag(r) @ (U - Ur)
        # Define dynamics and cost dictionaries.
        dynamics = {'dyn_eqn': X_dot, 'obs_eqn': Y, 'vars': {'X': X, 'U': U}}
        cost = {
            'cost_func': cost_func,
            'vars': {
                'X': X,
                'U': U,
                'Xr': Xr,
                'Ur': Ur,
                'Q': q,
                'R': r
            }
        }
        # Additional params to cache
        params = {
            # prior inertial properties
            'quad_mass': self.m,
            'quad_Iyy': self.Iyy,
            'quad_Ixx': self.Ixx if 'Ixx' in locals() else None,
            'quad_Izz': self.Izz if 'Izz' in locals() else None,
            # equilibrium point for linearization
            'X_EQ': self.X_EQ,
            'U_EQ': self.U_EQ,
        }

        # Setup symbolic model.
        self.symbolic = SymbolicModel(dynamics=dynamics, cost=cost, dt=self.dt)
