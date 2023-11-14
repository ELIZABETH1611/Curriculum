import numpy as np
import torch
from pathlib import Path
from mushroom_rl.core import Core
from experiments.train_agent import get_currot
from examples.rl.atacom_agent_wrapper import ATACOMAgent
from experiments.environments.planar3dof import AirHockeyPosition
from experiments.environments.air_hockey_wrapper import AirHockeyWrapper, AirHockeyCurriculum
from experiments.environments.boundary_distributions import TargetBoundarySampler


class DummySampler(AirHockeyCurriculum):

    def __init__(self, contexts):
        self.contexts = contexts
        self.count = 0

    def __call__(self, *args, **kwargs):
        tmp = self.contexts[self.count]
        self.count += 1
        return tmp.numpy()

    def update(self, contexts: np.ndarray, successes: np.ndarray):
        pass


def main():
    currot = get_currot(0.5, 0.2)
    currot.currot.load(Path(
        "/home/pascal/PycharmProjects/air_hockey_v2/air_hockey_challenge/experiments/atacom_planar_sparse_ap_0.0/currot_learner/seed_0_delta_0.6/curriculum_50"))

    success_samples = currot.currot.context_transform.color(currot.currot.success_buffer.get_contexts(1000))
    puck_velocities = torch.linalg.norm(success_samples[:, 4:], dim=-1)
    high_vel_contexts = success_samples[torch.argsort(puck_velocities)[-10:], :]
    high_vel_contexts = torch.repeat_interleave(high_vel_contexts, 2, dim=0)
    high_vel_contexts[1::2, 4:] = 0.

    # DummySampler(high_vel_contexts)
    env = AirHockeyWrapper(AirHockeyPosition, curriculum=TargetBoundarySampler(), action_penalty_scale=0.,
                           sparse=True)
    agent = ATACOMAgent.load_agent(
        "/home/pascal/PycharmProjects/air_hockey_v2/air_hockey_challenge/experiments/atacom_planar_sparse_ap_0.0/currot_learner/seed_0_delta_0.6/agent_50",
        env.env_info)

    core = Core(agent, env)

    core.evaluate(n_episodes=20, render=True)


if __name__ == "__main__":
    main()
