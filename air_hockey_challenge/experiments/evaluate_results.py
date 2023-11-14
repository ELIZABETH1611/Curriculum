import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from experiments.environments.boundary_distributions import TargetBoundarySampler
from experiments.train_agent import get_currot


def main():
    task_sampler = get_currot(0.5, 0.2)

    task_sampler.currot.load(Path(
        "/home/pascal/PycharmProjects/air_hockey_v2/air_hockey_challenge/experiments/atacom_planar_sparse_ap_0.0/currot_learner/seed_0_delta_0.6/curriculum_50"))

    task_distribution = task_sampler.currot.context_transform.color(
        task_sampler.currot.success_buffer.get_contexts(1000))
    plt.scatter(task_distribution[:, 0], task_distribution[:, 1], color="C0")
    plt.scatter(task_distribution[:, 2], task_distribution[:, 3], color="C1")
    plt.quiver(task_distribution[:, 2], task_distribution[:, 3], task_distribution[:, 4], task_distribution[:, 5],
               scale=1)
    plt.show()

    target_tasks = TargetBoundarySampler()(1000, concatenated=True)
    plt.scatter(target_tasks[:, 0], target_tasks[:, 1], color="C0")
    plt.scatter(target_tasks[:, 2], target_tasks[:, 3], color="C1")
    plt.quiver(target_tasks[:, 2], target_tasks[:, 3], target_tasks[:, 4], target_tasks[:, 5], scale=1)
    plt.show()


if __name__ == "__main__":
    main()
