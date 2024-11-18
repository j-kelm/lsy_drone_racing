import time
from copy import deepcopy
from termcolor import colored

import casadi as cs
import numpy as np

from lsy_drone_racing.control.mpc.mpc_utils import rk_discrete


class MPC:
    def __init__(self,
                 model,
                 horizon: int = 5,
                 q_mpc: list = [5],
                 r_mpc: list = [0.01],
                 warmstart: bool = True,
                 soft_penalty: float = 1e3,
                 constraint_tol: float = 1e-6,
                 # runner args
                 # shared/base args
                 output_dir: str = 'results/temp',
                 use_gpu: bool = False,
                 seed: int = 0,
                 solver: str = 'ipopt',
                 max_iter: int = 1000,
                 max_wall_time: float = 1.0e+20,
                 horizon_skip: int = 1,
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
            constraint_tol (float): Tolerance to add the constraint as sometimes solvers are not exact.
            output_dir (str): output directory to write logs and results.
            additional_constraints (list): List of additional constraints
            use_gpu (bool): False (use cpu) True (use cuda).
            seed (int): random seed.
        '''

        for k, v in locals().items():
            if k != 'self' and k != 'kwargs' and '__' not in k:
                self.__dict__.update({k: v})

        self.state_constraints = model.state_constraints
        self.input_constraints = model.input_constraints

        self.state_constraints_soft = model.state_constraints_soft
        self.input_constraints_soft = model.input_constraints_soft

        # Model parameters
        self.model = model.symbolic
        self.X_EQ = model.X_EQ
        self.U_EQ = model.U_EQ
        self.dt = self.model.dt
        self.T = horizon
        self.q = np.repeat(np.array(q_mpc)[:, np.newaxis], self.T+1, axis=1)
        self.r = np.repeat(np.array(r_mpc)[:, np.newaxis], self.T+1, axis=1)

        self.constraint_tol = constraint_tol
        self.soft_penalty = soft_penalty
        self.warmstart = warmstart
        self.horizon_skip = horizon_skip

        self.solver = solver
        self.max_iter = max_iter
        self.max_wall_time = max_wall_time

        self.set_dynamics_func()
        self.setup_optimizer(self.solver)

        self.reset()


    def close(self):
        '''Cleans up resources.'''
        pass

    def reset(self):
        print(colored('Resetting MPC', 'green'))
        '''Prepares for training or evaluation.'''
        # Dynamics model.
        # self.set_dynamics_func()
        # CasADi optimizer.
        # self.setup_optimizer(self.solver)
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

    def setup_optimizer(self, solver='ipopt', max_iter=None, max_wall_time=None):
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
        # q cost vector over horizon
        q = opti.parameter(nx, T + 1)
        # q cost vector over horizon
        r = opti.parameter(nu, T + 1)

        # Add slack variables
        state_slack = opti.variable(len(self.state_constraints_soft))
        input_slack = opti.variable(len(self.input_constraints_soft))

        # cost (cumulative)
        cost = 0
        cost_func = self.model.loss
        for i in range(T):
            # Can ignore the first state cost since fist x_var == x_init.
            cost += cost_func(x=x_var[:, i],
                              u=u_var[:, i],
                              Xr=x_ref[:, i],
                              Ur=np.zeros((nu, 1)),
                              Q=q[:, i],
                              R=r[:, i])['l']
        # Terminal cost.
        cost += cost_func(x=x_var[:, -1],
                          u=np.zeros((nu, 1)),
                          Xr=x_ref[:, -1],
                          Ur=np.zeros((nu, 1)),
                          Q=q[:, -1],
                          R=r[:, -1])['l']

        # Constraints
        for i in range(self.T):
            # Dynamics constraints.
            next_state = self.dynamics_func(x0=x_var[:, i], p=u_var[:, i])['xf']
            opti.subject_to(x_var[:, i + 1] == next_state)

            # hard constraints
            for sc_i, state_constraint in enumerate(self.state_constraints):
                opti.subject_to(state_constraint(x_var[:, i]) < -self.constraint_tol)
            for ic_i, input_constraint in enumerate(self.input_constraints):
                opti.subject_to(input_constraint(u_var[:, i]) < -self.constraint_tol)

            # soft constraints
            for sc_i, state_constraint in enumerate(self.state_constraints_soft):
                opti.subject_to(state_constraint(x_var[:, i]) <= state_slack[sc_i])
                cost += self.soft_penalty * state_slack[sc_i]**2
                opti.subject_to(state_slack[sc_i] >= 0)
            for ic_i, input_constraint in enumerate(self.input_constraints_soft):
                opti.subject_to(input_constraint(u_var[:, i]) <= input_slack[ic_i])
                cost += self.soft_penalty * input_slack[ic_i]**2
                opti.subject_to(input_slack[ic_i] >= 0)

        # Final state constraints.
        for sc_i, state_constraint in enumerate(self.state_constraints):
            opti.subject_to(state_constraint(x_var[:, -1]) <= -self.constraint_tol)
        for sc_i, state_constraint in enumerate(self.state_constraints_soft):
            opti.subject_to(state_constraint(x_var[:, -1]) <= state_slack[sc_i])
            cost += self.soft_penalty * state_slack[sc_i] ** 2
            opti.subject_to(state_slack[sc_i] >= 0)

        # initial condition constraints
        opti.subject_to(x_var[:, 0] == x_init)

        opti.minimize(cost)

        # Create solver
        jit_options = {"flags": ["-Ofast", "-march=native"], "verbose": True}

        opts = {'expand': True,
                'error_on_fail': False,
                'ipopt.print_level':0,
                'ipopt.print_timing_statistics': 'yes',
                'print_time':0,
                'record_time': 1,
                'jit': False,
                'compiler': 'shell',
                'jit_options': jit_options,
        }
        if max_iter is not None:
            opts['ipopt.max_iter'] = max_iter
        if max_wall_time is not None:
            opts['ipopt.max_wall_time'] = max_wall_time

        opti.solver(solver, opts)

        self.opti_dict = {
            'opti': opti,
            'x_var': x_var,
            'u_var': u_var,
            'x_init': x_init,
            'x_ref': x_ref,
            'cost': cost,
            'q': q,
            'r': r,
        }


    def select_action(self,
                      obs,
                      ref,
                      info: dict=None,
                      force_warm_start: bool=False,
                      ):
        '''Solves nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            ref (ndarray): Current state reference to track
            info (dict): Current info containing the reference, warmstart info, weighted cost matrices
            force_warm_start (bool): Force use of reference/provided guess for warm-starting

        Returns:
            action (ndarray): Input/action to the task/env.
        '''
        time_before = time.perf_counter()

        opti_dict = self.opti_dict
        opti = opti_dict['opti']
        x_var = opti_dict['x_var']  # optimization variables
        u_var = opti_dict['u_var']  # optimization variables
        x_init = opti_dict['x_init']  # initial state
        x_ref = opti_dict['x_ref']  # reference state/trajectory
        q = opti_dict['q']  # time dependant state cost matrix
        r = opti_dict['r']  # time dependant input cost matrix

        # Assign the initial state.
        opti.set_value(x_init, obs)
        # Assign reference trajectory within horizon.
        goal_states = self.to_horizon(goal_states=ref)
        opti.set_value(x_ref, goal_states)

        if "q" in info and info['q'] is not None:
            # use provided cost matrix
            opti.set_value(q, info['q'])
        else:
            # use default cost matrix
            opti.set_value(q, self.q)

        if "r" in info and info['r'] is not None:
            # use provided cost matrix
            opti.set_value(r, info['r'])
        else:
            # use default cost matrix
            opti.set_value(r, self.r)

        # check for warm start solution
        if force_warm_start or (self.x_prev is None and self.u_prev is None):
            if self.warmstart:
                if 'x_guess' in info and info['x_guess'] is not None:
                    print(colored(f'setting initial state guess from info', 'green'))
                    x_guess = info['x_guess']
                else:
                    print(colored(f'setting initial state guess from reference', 'green'))
                    x_guess = goal_states

                if 'u_guess' in info and info['u_guess'] is not None:
                    print(colored(f'setting initial input guess from info', 'green'))
                    u_guess = info['u_guess']
                else:
                    print(colored(f'setting initial input guess from reference', 'green'))
                    u_guess = np.tile(np.expand_dims(np.zeros(4), axis=1), (1, self.T))

                opti.set_initial(x_var, x_guess)
                opti.set_initial(u_var, u_guess)

        elif self.warmstart:
            # shift previous solutions by 1 step
            x_guess = deepcopy(self.x_prev)
            u_guess = deepcopy(self.u_prev)
            x_guess[:, :-self.horizon_skip] = x_guess[:, self.horizon_skip:]
            u_guess[:-self.horizon_skip] = u_guess[self.horizon_skip:]
            opti.set_initial(x_var, x_guess)
            opti.set_initial(u_var, u_guess)

        # Solve the optimization problem
        try:
            sol = opti.solve()
            x_val, u_val = sol.value(x_var), sol.value(u_var)
        except RuntimeError as e:
            status = opti.return_status()
            if status == 'NonIpopt_Exception_Thrown':
                raise e
            else:
                print(colored(f'Infeasible MPC Problem: {status}', 'red'))
                x_val, u_val = opti.debug.value(x_var), opti.debug.value(u_var)

        self.x_prev = x_val
        self.u_prev = u_val
        self.results_dict['horizon_states'].append(deepcopy(self.x_prev))
        self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev))
        y = np.array(self.model.g_func(x=self.x_prev[:, 1:],
                                                u=self.u_prev)['g'])
        self.results_dict['horizon_outputs'].append(deepcopy(y))
        self.results_dict['horizon_references'].append(deepcopy(goal_states))

        stats = opti.stats()
        self.results_dict['t_wall'].append(stats['t_wall_total'])
        self.results_dict['solution_found'].append(stats['success'])
        self.results_dict['iter_count'].append(stats['iter_count'])
        self.results_dict['obj'].append(stats['iterations']['obj'][-1])

        # Take the first action from the solved action sequence.
        if u_val.ndim > 1:
            actions = np.array(u_val[:, 0:])
            states = np.array(x_val[:, 0:])
            outputs = np.array(y[:, 0:])
        else:
            actions = np.array([u_val[0:]])
            states = np.array([x_val[0:]])
            outputs = np.array([y[0:]])

        time_after = time.perf_counter()
        print('MPC select_action time: ', time_after - time_before)
        return actions, states, outputs

    def to_horizon(self, goal_states):
        """Constructs reference states along mpc horizon. (nx, T+1)."""
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
                             'horizon_outputs': [],
                             'horizon_references': [],
                             'solution_found': [],
                             'frames': [],
                             'state_mse': [],
                             'common_cost': [],
                             'state': [],
                             'state_error': [],
                             't_wall': [],
                             'iter_count': [],
                             'obj': [],
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