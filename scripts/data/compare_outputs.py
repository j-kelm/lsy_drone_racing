import numpy as np
import matplotlib.pyplot as plt

from lsy_drone_racing.control.utils import to_global_action
from lsy_drone_racing.control.diffusion_controller import Controller



if __name__ == "__main__":
    n_samples = 1

    data = np.load("output/race_data.npz", allow_pickle=True)
    obs, actions = data['obs'], data['actions']
    local_obs, local_actions = data['local_obs'], data['local_actions']

    rng = np.random.default_rng()
    sample_index = rng.choice(len(obs), size=n_samples, replace=False)[0]

    # normalize local obs to find most similar
    local_obs_norm = (local_obs - local_obs.mean(axis=0))/local_obs.std(axis=0)

    # find most similar observation sample
    difference = np.linalg.norm(local_obs_norm-local_obs_norm[sample_index], axis=1)
    difference[sample_index] = np.inf
    closest_index = np.argmin(difference)

    # sample some diffusion actions
    diffusion_controller = Controller(None, None)
    diffusion_actions = diffusion_controller.sample_actions(local_obs[sample_index], samples=5)

    fig, axs = plt.subplots(actions.shape[1], sharex=True)
    fig.suptitle(f'Sample Index: {sample_index} | Closest Sample Index: {closest_index} | Closest Sample NRMSE: {difference[closest_index]:.3f}')

    action_labels = ['$x$', '$y$', '$z$',
                     '$v_x$', '$v_y$', '$v_z$',
                     '$a_x$', '$a_y$', '$a_z$',
                     'yaw',
                     '$p$', '$q$', '$r$']

    action_units = ['$m$', '$m$', '$m$',
                    '$\\frac{m}{s}$', '$\\frac{m}{s}$', '$\\frac{m}{s}$',
                    '$\\frac{m}{s^2}$', '$\\frac{m}{s^2}$', '$\\frac{m}{s^2}$',
                    '$rad$',
                    '$\\frac{rad}{s}$', '$\\frac{rad}{s}$', '$\\frac{rad}{s}$']

    for i, ax in enumerate(axs):
        ax.set_ylabel(action_units[i])
        ax.set_title(action_labels[i])
        ax.plot(local_actions[closest_index, i], color='r', linestyle='-.', label=None if i else 'closest')
        for j, action in enumerate(diffusion_actions):
            ax.plot(action[i], linestyle='dashed', alpha=0.75, label=None if i or j else 'diffusion')
        ax.plot(local_actions[sample_index, i], color='g', label=None if i else 'MPC')

    axs[-1].set_xlabel('time step')
    fig.legend(loc='lower right')
    plt.show()