import os
import copy
import pickle
import numpy as np
from experiments.environments.air_hockey_wrapper import AirHockeyCurriculum


def get_virtual_boundaries(goal: np.ndarray, table_length: float):
    # We automatically compute the boundary from the goal position. We do this such that we can easily compute integrate
    # CurrOT. The boundary computation is based on the interpolation between the minimum and maximum size of the
    # boundaries as defined in the boundary_distributions.py script
    easy_x_pos = EasyBoundarySampler.RECTANGLE_X + EasyBoundarySampler.RECTANGLE_HEIGHT
    easy_y_pos = 0.5 * EasyBoundarySampler.RECTANGLE_WIDTH
    hard_x_pos = TargetBoundarySampler.RECTANGLE_X + TargetBoundarySampler.RECTANGLE_HEIGHT
    hard_y_pos = 0.5 * TargetBoundarySampler.RECTANGLE_WIDTH

    x_scale = (goal[..., 0] - easy_x_pos) / (hard_x_pos - easy_x_pos)
    y_scale = (np.abs(goal[..., 1]) - easy_y_pos) / (hard_y_pos - easy_y_pos)
    max_scale = np.maximum(x_scale, y_scale)

    half_width = easy_y_pos + max_scale * (hard_y_pos - easy_y_pos)
    return np.stack([-(table_length / 2) * np.ones_like(goal[..., 0]), -half_width], axis=-1), \
        np.stack([easy_x_pos + max_scale * (hard_x_pos - easy_x_pos), half_width], axis=-1)


def sample_rectangle(x_offset: float, width: float, height: float, n: int):
    # Total length of the rectangle is 2 * height + width
    pos = np.random.uniform(0, 2 * height + width, size=(n,))

    x_pos = np.where(pos < height, pos,
                     np.where(pos < height + width, height, 2 * height + width - pos))
    y_pos = np.where(pos < height, -0.5 * width,
                     np.where(pos < height + width, pos - height - 0.5 * width, 0.5 * width))
    return np.stack((x_offset + x_pos, y_pos), axis=-1)


class BoundarySampler(AirHockeyCurriculum):

    def __init__(self, puck_bounds, rectangle_x, rectangle_width, rectangle_height):
        self.puck_bounds = copy.deepcopy(puck_bounds)
        self.rectangle_x = copy.deepcopy(rectangle_x)
        self.rectangle_width = copy.deepcopy(rectangle_width)
        self.rectangle_height = copy.deepcopy(rectangle_height)

    def __call__(self, n: int = 1, concatenated: bool = False):
        # Total length of the rectangle is 2 * height + width
        goal_pos = sample_rectangle(self.rectangle_x, self.rectangle_width, self.rectangle_height, n)
        puck_pos = np.random.uniform(self.puck_bounds[0], self.puck_bounds[1], size=(n, 2))
        puck_vel = np.zeros((n, 2))

        if concatenated:
            return np.squeeze(np.concatenate((goal_pos, puck_pos, puck_vel), axis=-1))
        else:
            return np.squeeze(goal_pos), np.squeeze(puck_pos), np.squeeze(puck_vel)

    def update(self, contexts: np.ndarray, successes: np.ndarray):
        # No update here
        pass


class TargetBoundarySampler(BoundarySampler):
    PUCK_BOUNDS = (np.array([-0.65, -0.35]), np.array([-0.25, 0.35]))
    RECTANGLE_X = 0.
    RECTANGLE_WIDTH = 1.038  # Corresponds to self.env_info['table']['width']
    RECTANGLE_HEIGHT = 0.974  # Corresponds to self.env_info['table']['height']

    def __init__(self):
        super().__init__(self.PUCK_BOUNDS, self.RECTANGLE_X, self.RECTANGLE_WIDTH, self.RECTANGLE_HEIGHT)


class EasyBoundarySampler(BoundarySampler):
    PUCK_BOUNDS = (np.array([-0.7, -0.02]), np.array([-0.68, 0.02]))
    RECTANGLE_X = -0.7
    RECTANGLE_WIDTH = 0.15
    RECTANGLE_HEIGHT = 0.1

    def __init__(self):
        super().__init__(self.PUCK_BOUNDS, self.RECTANGLE_X, self.RECTANGLE_WIDTH, self.RECTANGLE_HEIGHT)


class BoundaryCurriculum(AirHockeyCurriculum):
    def __init__(self, n_steps: int, performance_threshold: float):
        self.step = 0
        self.n_steps = n_steps
        self.performance_threshold = performance_threshold
        lbs = np.linspace(EasyBoundarySampler.PUCK_BOUNDS[0], TargetBoundarySampler.PUCK_BOUNDS[0], n_steps)
        ubs = np.linspace(EasyBoundarySampler.PUCK_BOUNDS[1], TargetBoundarySampler.PUCK_BOUNDS[1], n_steps)
        self.puck_bounds = [(lb, ub) for lb, ub in zip(lbs, ubs)]
        self.rectangle_x = list(
            np.linspace(EasyBoundarySampler.RECTANGLE_X, TargetBoundarySampler.RECTANGLE_X, n_steps))
        self.rectangle_width = list(
            np.linspace(EasyBoundarySampler.RECTANGLE_WIDTH, TargetBoundarySampler.RECTANGLE_WIDTH, n_steps))
        self.rectangle_height = list(
            np.linspace(EasyBoundarySampler.RECTANGLE_HEIGHT, TargetBoundarySampler.RECTANGLE_HEIGHT, n_steps))

    def update(self, contexts, performances):
        if np.mean(performances) > self.performance_threshold:
            self.step += 1
            if self.step >= self.n_steps:
                self.step = self.n_steps - 1
        print(f"Curriculum Step: {self.step}")

    def __call__(self, n: int = 1, concatenated: bool = False):
        goal_pos = sample_rectangle(self.rectangle_x[self.step], self.rectangle_width[self.step],
                                    self.rectangle_height[self.step], n)
        puck_pos = np.random.uniform(self.puck_bounds[self.step][0], self.puck_bounds[self.step][1], size=(n, 2))
        puck_vel = np.zeros((n, 2))

        if concatenated:
            return np.squeeze(np.concatenate((goal_pos, puck_pos, puck_vel), axis=-1))
        else:
            return np.squeeze(goal_pos), np.squeeze(puck_pos), np.squeeze(puck_vel)

    def save(self, save_path):
        with open(os.path.join(save_path, "teacher.pkl"), "wb") as f:
            pickle.dump(self.step, f)
