import time
from copy import deepcopy
from termcolor import colored

import casadi as cs
import numpy as np

from examples.mpc_utils import get_cost_weight_matrix, rk_discrete
from examples.constraints import GENERAL_CONSTRAINTS, create_constraint_list, ConstraintList

def reset_constraints(constraints):
    '''Set up the constraints list.

    Args:
        constraints (list): List of constraints controller is subject too.
    '''

    constraints_list = ConstraintList(constraints)
    state_constraints_sym = constraints_list.get_state_constraint_symbolic_models()
    input_constraints_sym = constraints_list.get_input_constraint_symbolic_models()
    if len(constraints_list.input_state_constraints) > 0:
        raise NotImplementedError('[Error] Cannot handle combined state input constraints yet.')
    return constraints_list, state_constraints_sym, input_constraints_sym


class MPC:
    def __init__(self,
                 model,
                 horizon: int = 5,
                 q_mpc: list = [5],
                 r_mpc: list = [0.01],
                 warmstart: bool = True,
                 soft_constraints: bool = False,
                 soft_penalty: float = 10000,
                 terminate_run_on_done: bool = True,
                 constraint_tol: float = 1e-6,
                 # runner args
                 # shared/base args
                 output_dir: str = 'results/temp',
                 additional_constraints: list = None,
                 use_gpu: bool = False,
                 seed: int = 0,
                 compute_ipopt_initial_guess: bool = False,
                 init_solver: str = 'ipopt',
                 solver: str = 'ipopt',
                 **kwargs
                 ):

        '''Creates task and controller.

        Args:
            model (Model): Instance for MPC model.
            horizon (int): mpc planning horizon.
            q_mpc (list): diagonals of state cost weight.
            r_mpc (list): diagonals of input/action cost weight.
            warmstart (bool): if to initialize from previous iteration.
            soft_constraints (bool): Formulate the constraints as soft constraints.
            terminate_run_on_done (bool): Terminate the run when the environment returns done or not.
            constraint_tol (float): Tolerance to add the constraint as sometimes solvers are not exact.
            output_dir (str): output directory to write logs and results.
            additional_constraints (list): List of additional constraints
            use_gpu (bool): False (use cpu) True (use cuda).
            seed (int): random seed.
        '''

        for k, v in locals().items():
            if k != 'self' and k != 'kwargs' and '__' not in k:
                self.__dict__.update({k: v})

        if additional_constraints is not None:
            additional_ConstraintsList = create_constraint_list(additional_constraints,
                                                                GENERAL_CONSTRAINTS,
                                                                self.env)
            self.additional_constraints = additional_ConstraintsList.constraints
            self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(model.constraints + self.additional_constraints)
        else:
            self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(model.constraints)
            self.additional_constraints = []

        # Model parameters
        self.model = model.symbolic
        self.X_EQ = model.X_EQ
        self.U_EQ = model.U_EQ
        self.dt = self.model.dt
        self.T = horizon
        self.Q = get_cost_weight_matrix(self.q_mpc, self.model.nx)
        self.R = get_cost_weight_matrix(self.r_mpc, self.model.nu)

        self.constraint_tol = constraint_tol
        self.soft_constraints = soft_constraints
        self.soft_penalty = soft_penalty
        self.warmstart = warmstart
        self.terminate_run_on_done = terminate_run_on_done

        self.init_solver = init_solver
        self.solver = solver
        self.compute_ipopt_initial_guess = compute_ipopt_initial_guess

        self.reset()

    def add_constraints(self,
                        constraints
                        ):
        '''Add the constraints (from a list) to the system.

        Args:
            constraints (list): List of constraints controller is subject too.
        '''
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(constraints + self.constraints.constraints)

    def remove_constraints(self,
                           constraints
                           ):
        '''Remove constraints from the current constraint list.

        Args:
            constraints (list): list of constraints to be removed.
        '''
        old_constraints_list = self.constraints.constraints
        for constraint in constraints:
            assert constraint in self.constraints.constraints, \
                ValueError('This constraint is not in the current list of constraints')
            old_constraints_list.remove(constraint)
        self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(old_constraints_list)

    def close(self):
        '''Cleans up resources.'''
        pass

    def reset(self):
        print(colored('Resetting MPC', 'green'))
        '''Prepares for training or evaluation.'''
        # Dynamics model.
        self.set_dynamics_func()
        # CasADi optimizer.
        self.setup_optimizer(self.solver)
        # Previously solved states & inputs, useful for warm start.
        self.x_prev = None
        self.u_prev = None

        self.setup_results_dict()

    def set_dynamics_func(self):
        '''Updates symbolic dynamics with actual control frequency.'''
        self.dynamics_func = rk_discrete(self.model.fc_func,
                                         self.model.nx,
                                         self.model.nu,
                                         self.dt)


    def compute_initial_guess(self, init_state, goal_states):
        time_before = time.time()
        '''Use IPOPT to get an initial guess of the '''
        print(colored('Computing initial guess', 'green'))
        self.setup_optimizer(solver=self.init_solver)
        opti_dict = self.opti_dict
        opti = opti_dict['opti']
        x_var = opti_dict['x_var']  # optimization variables
        u_var = opti_dict['u_var']  # optimization variables
        x_init = opti_dict['x_init']  # initial state
        x_ref = opti_dict['x_ref']  # reference state/trajectory

        # Assign the initial state.
        opti.set_value(x_init, init_state)  # initial state should have dim (nx,)
        # Assign reference trajectory within horizon.
        goal_states = self.to_horizon(goal_states=goal_states)
        opti.set_value(x_ref, goal_states)

        # Solve the optimization problem.
        try:
            sol = opti.solve()
            x_val, u_val = sol.value(x_var), sol.value(u_var)
        except RuntimeError:
            print(colored('Warm-starting fails', 'red'))
            x_val, u_val = opti.debug.value(x_var), opti.debug.value(u_var)

        x_guess = x_val
        u_guess = u_val
        self.x_prev = x_guess
        self.u_prev = u_guess

        # set the solver back
        self.setup_optimizer(solver=self.solver)

        time_after = time.time()
        print('MPC _compute_initial_guess time: ', time_after - time_before)

        return x_guess, u_guess

    def setup_optimizer(self, solver='qrsqp'):
        '''Sets up nonlinear optimization problem.'''
        print(colored(f'Setting up optimizer with {solver}', 'green'))
        nx, nu = self.model.nx, self.model.nu
        T = self.T
        # Define optimizer and variables.
        opti = cs.Opti()
        # States.
        x_var = opti.variable(nx, T + 1)
        # Inputs.
        u_var = opti.variable(nu, T)
        # Initial state.
        x_init = opti.parameter(nx, 1)
        # Reference (equilibrium point or trajectory, last step for terminal cost).
        x_ref = opti.parameter(nx, T + 1)
        # Add slack variables
        state_slack = opti.variable(len(self.state_constraints_sym))
        input_slack = opti.variable(len(self.input_constraints_sym))

        # cost (cumulative)
        cost = 0
        cost_func = self.model.loss
        for i in range(T):
            # Can ignore the first state cost since fist x_var == x_init.
            cost += cost_func(x=x_var[:, i],
                              u=u_var[:, i],
                              Xr=x_ref[:, i],
                              Ur=np.zeros((nu, 1)),
                              Q=self.Q,
                              R=self.R)['l']
        # Terminal cost.
        cost += cost_func(x=x_var[:, -1],
                          u=np.zeros((nu, 1)),
                          Xr=x_ref[:, -1],
                          Ur=np.zeros((nu, 1)),
                          Q=self.Q,
                          R=self.R)['l']

        # Constraints
        for i in range(self.T):
            # Dynamics constraints.
            next_state = self.dynamics_func(x0=x_var[:, i], p=u_var[:, i])['xf']
            opti.subject_to(x_var[:, i + 1] == next_state)

            for sc_i, state_constraint in enumerate(self.state_constraints_sym):
                if self.soft_constraints:
                    opti.subject_to(state_constraint(x_var[:, i]) <= state_slack[sc_i])
                    cost += self.soft_penalty * state_slack[sc_i]**2
                    opti.subject_to(state_slack[sc_i] >= 0)
                else:
                    opti.subject_to(state_constraint(x_var[:, i]) < -self.constraint_tol)
            for ic_i, input_constraint in enumerate(self.input_constraints_sym):
                if self.soft_constraints:
                    opti.subject_to(input_constraint(u_var[:, i]) <= input_slack[ic_i])
                    cost += self.soft_penalty * input_slack[ic_i]**2
                    opti.subject_to(input_slack[ic_i] >= 0)
                else:
                    opti.subject_to(input_constraint(u_var[:, i]) < -self.constraint_tol)

        # TODO: Move constraints to right place
        opti.subject_to(opti.bounded(0.03, u_var, 0.145))  # thrust limits 0.03 - 0.145, includes safety margin
        opti.subject_to(opti.bounded(0.01, x_var[2, :], 2.5))  # room limit

        # Final state constraints.
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            if self.soft_constraints:
                opti.subject_to(state_constraint(x_var[:, -1]) <= state_slack[sc_i])
                cost += self.soft_penalty * state_slack[sc_i] ** 2
                opti.subject_to(state_slack[sc_i] >= 0)
            else:
                opti.subject_to(state_constraint(x_var[:, -1]) <= -self.constraint_tol)
        # initial condition constraints
        opti.subject_to(x_var[:, 0] == x_init)

        opti.minimize(cost)
        # Create solver
        opts = {'expand': True, 'error_on_fail': False}
        opti.solver(solver, opts)

        self.opti_dict = {
            'opti': opti,
            'x_var': x_var,
            'u_var': u_var,
            'x_init': x_init,
            'x_ref': x_ref,
            'cost': cost
        }

    def select_action(self,
                      obs,
                      info=None
                      ):
        '''Solves nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            info (dict): Current info

        Returns:
            action (ndarray): Input/action to the task/env.
        '''
        time_before = time.time()

        opti_dict = self.opti_dict
        opti = opti_dict['opti']
        x_var = opti_dict['x_var']  # optimization variables
        u_var = opti_dict['u_var']  # optimization variables
        x_init = opti_dict['x_init']  # initial state
        x_ref = opti_dict['x_ref']  # reference state/trajectory

        # Assign the initial state.
        opti.set_value(x_init, obs)
        # Assign reference trajectory within horizon.
        goal_states = self.to_horizon(goal_states=info["ref"])
        opti.set_value(x_ref, goal_states)

        # check for warm start solution
        if self.x_prev is None and self.u_prev is None:
            if self.compute_ipopt_initial_guess:
                print(colored(f'computing initial guess with {self.init_solver}', 'green'))
                x_guess, u_guess = self.compute_initial_guess(obs, goal_states)
                opti.set_initial(x_var, x_guess)
                opti.set_initial(u_var, u_guess) # Initial guess for optimization problem.
            elif self.warmstart:
                print(colored(f'setting initial guess from reference', 'green'))
                x_guess = goal_states
                u_guess = np.tile(np.expand_dims(self.U_EQ, axis=1), (1, self.T))
                opti.set_initial(x_var, x_guess)
                opti.set_initial(u_var, u_guess)

        elif self.warmstart:
            # shift previous solutions by 1 step
            x_guess = deepcopy(self.x_prev)
            u_guess = deepcopy(self.u_prev)
            x_guess[:, :-1] = x_guess[:, 1:]
            u_guess[:-1] = u_guess[1:]
            opti.set_initial(x_var, x_guess)
            opti.set_initial(u_var, u_guess)
        # Solve the optimization problem.
        try:
            sol = opti.solve()
            x_val, u_val = sol.value(x_var), sol.value(u_var)
        except RuntimeError:
            print(colored('Infeasible MPC Problem', 'red'))
            if self.solver == 'ipopt':
                x_val, u_val = opti.debug.value(x_var), opti.debug.value(u_var)
            elif self.solver == 'qrsqp':
                return_status = opti.return_status()
                print(f'Optimization failed with status: {return_status}')
                if return_status == 'unknown':
                    # self.terminate_loop = True
                    u_val = self.u_prev
                    x_val = self.x_prev
                    if u_val is None:
                        print('[WARN]: MPC Infeasible first step.')
                        u_val = u_guess
                        x_val = x_guess
                elif return_status == 'Maximum_Iterations_Exceeded':
                    self.terminate_loop = True
                    u_val = opti.debug.value(u_var)
                    x_val = opti.debug.value(x_var)
                elif return_status == 'Search_Direction_Becomes_Too_Small':
                    self.terminate_loop = True
                    u_val = opti.debug.value(u_var)
                    x_val = opti.debug.value(x_var)

        self.x_prev = x_val
        self.u_prev = u_val
        self.results_dict['horizon_states'].append(deepcopy(self.x_prev))
        self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev))
        self.results_dict['goal_states'].append(deepcopy(goal_states))
        if self.solver == 'ipopt':
            self.results_dict['t_wall'].append(opti.stats()['t_wall_total'])
        # Take the first action from the solved action sequence.
        if u_val.ndim > 1:
            action = u_val[:, 0]
            state = x_val[:, 1]
        else:
            action = np.array([u_val[0]])
            state = np.array([x_val[1]])
        self.prev_action = action
        self.prev_state = state
        time_after = time.time()
        print('MPC select_action time: ', time_after - time_before)
        return action, state

    def to_horizon(self, goal_states):
        '''Constructs reference states along mpc horizon.(nx, T+1).'''
        # Slice trajectory for horizon steps, if not long enough, repeat last state.
        start = 0
        end = min(self.T + 1, goal_states.shape[-1])
        remain = max(0, self.T + 1 - (end - start))
        goal_states = np.concatenate([
            goal_states[:, start:end],
            np.tile(goal_states[:, -1:], (1, remain))
        ], -1)

        return goal_states  # (nx, T+1).

    def setup_results_dict(self):
        '''Setup the results dictionary to store run information.'''
        self.results_dict = {'obs': [],
                             'reward': [],
                             'done': [],
                             'info': [],
                             'action': [],
                             'horizon_inputs': [],
                             'horizon_states': [],
                             'goal_states': [],
                             'frames': [],
                             'state_mse': [],
                             'common_cost': [],
                             'state': [],
                             'state_error': [],
                             't_wall': []
                             }

    def reset_before_run(self, obs, info=None, env=None):
        '''Reinitialize just the controller before a new run.

        Args:
            obs (ndarray): The initial observation for the new run.
            info (dict): The first info of the new run.
            env (BenchmarkEnv): The environment to be used for the new run.
        '''
        self.reset()

    def wrap_sym(self, X):
        '''Wrap angle to [-pi, pi] when used in observation.

        Args:
            X (ndarray): The state to be wrapped.

        Returns:
            X_wrapped (ndarray): The wrapped state.
        '''
        X_wrapped = cs.fmod(X[0] + cs.pi, 2 * cs.pi) - cs.pi
        return X_wrapped