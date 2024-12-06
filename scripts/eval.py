"""Kaggle competition auto-submission script.

Note:
    Please do not alter this script or ask the course supervisors first!
"""

import logging
from pathlib import Path

from sim import simulate
from statistics import fmean

from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)


def main():
    """Run the simulation N times and save the results as 'submission.csv'."""
    n_runs = 100
    config = load_config(Path(__file__).parents[1] / "config/multi_modality.toml")
    ep_times = simulate(
        config="multi_modality.toml", controller=config.controller.file, n_runs=n_runs, gui=False
    )

    success_times = [x for x in ep_times if x is not None]
    success_rate = len(success_times)/len(ep_times)

    if len(success_times):
        success_mean = fmean(success_times)
    else:
        success_mean = -1

    logger.info(f"Average over {n_runs} runs. Success rate: {success_rate*100:.2f} % | Mean success time: {success_mean:.3f} s")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
