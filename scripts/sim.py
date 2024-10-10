"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config config/level0.toml

Look for instructions in `README.md` and `edit_this.py`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import pybullet as p

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from munch import Munch

    from lsy_drone_racing.envs.drone_racing_env import DroneRacingEnv


logger = logging.getLogger(__name__)


def simulate(
    config: str = "config/level0.toml",
    controller: str = "examples/trajectory_controller.py",
    n_runs: int = 1,
    gui: bool = True,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file.
        controller: The path to the controller module.
        n_runs: The number of episodes.
        gui: Enable/disable the simulation GUI.

    Returns:
        A list of episode times.
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(config))
    config.sim.gui = gui
    # Load the controller module
    path = Path(__file__).parents[1] / controller
    controller_cls = load_controller(path)  # This returns a class, not an instance
    # Create the racing environment
    env: DroneRacingEnv = gymnasium.make("DroneRacing-v0", config=config)

    ep_times = []
    for _ in range(n_runs):  # Run n_runs episodes with the controller
        done = False
        obs, info = env.reset()
        controller = controller_cls(obs, info)
        if gui:
            gui_timer = update_gui_timer(0.0, env.unwrapped.sim.pyb_client)
        i = 0

        while not done:
            t_start = time.time()
            curr_time = i / config.env.freq
            if gui:
                gui_timer = update_gui_timer(curr_time, env.unwrapped.sim.pyb_client, gui_timer)

            action = controller.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Update the controller internal state and models.
            controller.step_learn(action, obs, reward, terminated, truncated, info)
            # Add up reward, collisions

            # Synchronize the GUI.
            if config.sim.gui:
                if (elapsed := time.time() - t_start) < 1 / config.env.freq:
                    time.sleep(1 / config.env.freq - elapsed)
            i += 1

        # Learn after the episode if the controller supports it
        controller.episode_learn()  # Update the controller internal state and models.
        log_episode_stats(info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if info["target_gate"] == -1 else None)

    # Close the environment
    env.close()
    return ep_times


def update_gui_timer(t: float, client_id: int, g_id: int | None = None) -> int:
    """Update the timer in the GUI."""
    text = f"Ep. time: {t:.2f}s"
    if g_id is None:
        return p.addUserDebugText(text, textPosition=[0, 0, 1.5], physicsClientId=client_id)
    return p.addUserDebugText(
        text,
        textPosition=[0, 0, 1.5],
        textColorRGB=[1, 0, 0],
        lifeTime=0,
        textSize=1.5,
        parentObjectUniqueId=0,
        parentLinkIndex=-1,
        replaceItemUniqueId=g_id,
        physicsClientId=client_id,
    )


def log_episode_stats(info: dict, config: Munch, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = info["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
        gates_passed = len(config.env.track.gates)
    if info["collisions"]:
        termination = "Collision"
    elif info["target_gate"] == -1:
        termination = "Task completed"
    else:
        termination = "Unknown"
    logger.info(
        (
            f"Flight time (s): {curr_time}\n"
            f"Reason for termination: {termination}\n"
            f"Gates passed: {gates_passed}\n"
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(simulate)
