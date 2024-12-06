"""General MPC utility functions.

"""
import casadi as cs

states_for_obs = range(12)
outputs_for_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14]

def rk_discrete(f, n, m, dt):
    """Runge Kutta discretization for the function.

    Args:
        f (casadi function): Function to discretize.
        n (int): state dimensions.
        m (int): input dimension.
        dt (float): discretization time.

    Return:
        x_next (casadi function?):
    """
    X = cs.SX.sym('X', n)
    U = cs.SX.sym('U', m)
    # Runge-Kutta 4 integration
    k1 = f(X,         U)
    k2 = f(X+dt/2*k1, U)
    k3 = f(X+dt/2*k2, U)
    k4 = f(X+dt*k3,   U)
    x_next = X + dt/6*(k1+2*k2+2*k3+k4)
    rk_dyn = cs.Function('rk_f', [X, U], [x_next], ['x0', 'p'], ['xf'])

    return rk_dyn
