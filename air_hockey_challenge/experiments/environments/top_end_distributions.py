import os
import copy
import pickle
import numpy as np
from experiments.environments.air_hockey_wrapper import AirHockeyCurriculum


class TopEndSampler(AirHockeyCurriculum):

    def __init__(self, goal_bounds, puck_bounds):
        self.puck_bounds = copy.deepcopy(puck_bounds)
        self.goal_bounds = copy.deepcopy(goal_bounds)

    def __call__(self, n: int = 1, concatenated: bool = False):
        # Total length of the rectangle is 2 * height + width
        goal_pos = np.random.uniform(self.goal_bounds[0], self.goal_bounds[1], size=(n, 2))
        puck_pos = np.random.uniform(self.puck_bounds[0], self.puck_bounds[1], size=(n, 2))
        puck_vel = np.zeros((n, 2))

        # Check that the goal is at least 5 cm above the puck
        resample = np.linalg.norm(goal_pos - puck_pos, axis=-1) < 0.05
        while np.any(resample):
            puck_pos[resample] = np.random.uniform(self.puck_bounds[0], self.puck_bounds[1], size=(np.sum(resample), 2))
            resample = np.linalg.norm(goal_pos - puck_pos, axis=-1) < 0.05

        if concatenated:
            return np.squeeze(np.concatenate((goal_pos, puck_pos, puck_vel), axis=-1))
        else:
            return np.squeeze(goal_pos), np.squeeze(puck_pos), np.squeeze(puck_vel)

    def update(self, contexts: np.ndarray, successes: np.ndarray):
        # No update here
        pass


class TargetTopEndSampler(TopEndSampler):
    GOAL_BOUNDS = ([0.75, -0.35], [0.9, 0.35])
    PUCK_BOUNDS = ([-0.65, -0.35], [-0.25, 0.35])

    def __init__(self):
        super().__init__(self.GOAL_BOUNDS, self.PUCK_BOUNDS)


class EasyTopEndSampler(TopEndSampler):
    GOAL_BOUNDS = ([-0.60, -0.04], [-0.54, 0.04])
    PUCK_BOUNDS = ([-0.65, -0.02], [-0.60, 0.02])

    def __init__(self):
        super().__init__(self.GOAL_BOUNDS, self.PUCK_BOUNDS)


class TopEndCurriculum(AirHockeyCurriculum):
    def __init__(self, n_steps: int, performance_threshold: float):
        self.step = 0
        self.n_steps = n_steps
        self.performance_threshold = performance_threshold
        lbs = np.linspace(EasyTopEndSampler.PUCK_BOUNDS[0], TargetTopEndSampler.PUCK_BOUNDS[0], n_steps)
        ubs = np.linspace(EasyTopEndSampler.PUCK_BOUNDS[1], TargetTopEndSampler.PUCK_BOUNDS[1], n_steps)
        self.puck_bounds = [(lb, ub) for lb, ub in zip(lbs, ubs)]

        lbs = np.linspace(EasyTopEndSampler.GOAL_BOUNDS[0], TargetTopEndSampler.GOAL_BOUNDS[0], n_steps)
        ubs = np.linspace(EasyTopEndSampler.GOAL_BOUNDS[1], TargetTopEndSampler.GOAL_BOUNDS[1], n_steps)
        self.goal_bounds = [(lb, ub) for lb, ub in zip(lbs, ubs)]

    def update(self, contexts, performances):
        if np.mean(performances) > self.performance_threshold:
            self.step += 1
            if self.step >= self.n_steps:
                self.step = self.n_steps - 1
        print(f"Curriculum Step: {self.step}")

    def __call__(self, n: int = 1, concatenated: bool = False):
        # Total length of the rectangle is 2 * height + width
        goal_pos = np.random.uniform(self.goal_bounds[self.step][0], self.goal_bounds[self.step][1], size=(n, 2))
        puck_pos = np.random.uniform(self.puck_bounds[self.step][0], self.puck_bounds[self.step][1], size=(n, 2))
        puck_vel = np.zeros((n, 2))

        # Check that the goal is at least 5 cm above the puck
        resample = np.linalg.norm(goal_pos - puck_pos, axis=-1) < 0.05
        while np.any(resample):
            puck_pos[resample] = np.random.uniform(self.puck_bounds[self.step][0], self.puck_bounds[self.step][1],
                                                   size=(np.sum(resample), 2))
            resample = np.linalg.norm(goal_pos - puck_pos, axis=-1) < 0.05

        if concatenated:
            return np.squeeze(np.concatenate((goal_pos, puck_pos, puck_vel), axis=-1))
        else:
            return np.squeeze(goal_pos), np.squeeze(puck_pos), np.squeeze(puck_vel)

    def save(self, save_path):
        with open(os.path.join(save_path, "teacher.pkl"), "wb") as f:
            pickle.dump(self.step, f)
