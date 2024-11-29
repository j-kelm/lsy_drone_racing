import numpy as np
import matplotlib.pyplot as plt

from lsy_drone_racing.control.utils import to_global_action
from lsy_drone_racing.control.diffusion_controller import Controller

MAX_DIFFUSION_SAMPLES = 5

if __name__ == "__main__":
    n_samples = 1

    train_data = np.load("output/race_data.npz", allow_pickle=True)
    flight_data = np.load("output/logs/sim_diffusion.npz", allow_pickle=True)

    # obs, actions = data['obs'], data['actions']
    local_train_obs, train_actions = train_data['local_obs'], train_data['actions']
    local_flight_obs, flight_mpc_actions, flight_diffusion_actions = flight_data['states'], flight_data['mpc_actions'], flight_data['diffusion_actions']
    print('Loading done :D')

    # cut mpc actions to right length
    flight_mpc_actions = flight_mpc_actions[:, :, :flight_diffusion_actions.shape[-1]]
    train_actions = train_actions[:, :, :flight_diffusion_actions.shape[-1]]

    # normalize local obs to find most similar
    local_obs_mean = local_train_obs.mean(axis=0)
    local_obs_std = local_train_obs.std(axis=0)
    local_obs_std = local_obs_std + np.full_like(local_obs_std, 1.0e-8)
    local_train_obs_norm = (local_train_obs - local_obs_mean)/local_obs_std
    local_flight_obs_norm = (local_flight_obs - local_obs_mean)/local_obs_std

    action_labels = ['$x$', '$y$', '$z$',
                     '$v_x$', '$v_y$', '$v_z$',
                     '$a_x$', '$a_y$', '$a_z$',
                     'yaw',
                     '$p$', '$q$', '$r$']

    action_units = ['$m$', '$m$', '$m$',
                    '$\\frac{m}{s}$', '$\\frac{m}{s}$', '$\\frac{m}{s}$',
                    '$\\frac{m}{s^2}$', '$\\frac{m}{s^2}$', '$\\frac{m}{s^2}$',
                    'rad',
                    '$\\frac{rad}{s}$', '$\\frac{rad}{s}$', '$\\frac{rad}{s}$']

    # loop over all data points in flight data
    for obs, mpc_action, diffusion_action in zip(local_flight_obs_norm, flight_mpc_actions, flight_diffusion_actions):

        # find most similar observation sample
        difference = np.linalg.norm(local_train_obs_norm - obs, axis=1)
        closest_index = np.argmin(difference)
        print('Found most similar \\o/')

        fig, axs = plt.subplots(mpc_action.shape[0], sharex=True, figsize=(30, 20))
        fig.suptitle(f'Closest Sample Index: {closest_index} | Closest Sample NRMSE: {difference[closest_index]:.3f}')

        for i, ax in enumerate(axs):
            ax.set_title(action_labels[i])
            ax.set_ylabel(action_units[i], rotation=0)

            ax.plot(train_actions[closest_index, i], color='r', linestyle='-.', label=None if i else 'closest')
            for j, action in enumerate(diffusion_action[:MAX_DIFFUSION_SAMPLES]):
                ax.plot(action[i], linestyle='dashed', alpha=0.75, label=None if i or j else 'diffusion')
            ax.plot(mpc_action[i], color='g', label=None if i else 'MPC')

        axs[-1].set_xlabel('time step')
        fig.legend(loc='lower right')
        plt.show()