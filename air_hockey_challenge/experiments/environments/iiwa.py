import numpy as np
from experiments.environments import util
from air_hockey_challenge.environments.iiwas.env_single import AirHockeySingle
from air_hockey_challenge.environments.position_control_wrapper import PositionControlIIWA


class AirHockey(AirHockeySingle):
    """
    Class for the air hockey hitting task using CL.
    """

    def __init__(self, task_sampler, gamma=0.99, horizon=500, viewer_params={}, sparse: bool = False,
                 stop_on_all_boundaries: bool = True):
        """
        Constructor
        Args:
            moving_init(bool, False): If true, initialize the puck with inital velocity.
        """
        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

        self.sparse = sparse
        self.gamma = gamma

        self._internal_info = None

        self.goal = np.array([0.945, 0])
        self.task_sampler = task_sampler
        self.stop_on_all_boundaries = stop_on_all_boundaries

    def _modify_mdp_info(self, mdp_info):
        return util.modify_info_fn(AirHockey, self, mdp_info)

    def _modify_observation(self, obs):
        return util.modify_obs_fn(AirHockey, self, obs)

    def setup(self, state=None):
        util.setup_fn(AirHockey, self, state=state)

    def reward(self, state, action, next_state, absorbing):
        return util.reward_fn(self._internal_info, sparse=self.sparse, gamma=self.info.gamma)

    def is_absorbing(self, obs):
        if self.stop_on_all_boundaries:
            self._internal_info = util.all_absorbing_fn(self, obs)
        else:
            self._internal_info = util.top_absorbing_fn(self, obs)
        return self._internal_info["done"]

    def _create_info_dictionary(self, obs):
        '''
        Create dictionary to save when the task was accomplished or not
        '''

        return {"success_task": np.array([self._internal_info["success"]]),
                "intercepted": np.array([self._internal_info["puck_intercepted"]])}


class AirHockeyPosition(PositionControlIIWA, AirHockey):
    pass


if __name__ == '__main__':
    from experiments.environments.boundary_distributions import TargetBoundarySampler

    env = AirHockeyPosition(task_sampler=TargetBoundarySampler(), sparse=True)

    env.reset()
    env.render()
    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    while True:
        action = np.zeros((2, 7))  # np.random.uniform(-1, 1, 3)  # np.zeros(3)

        observation, reward, done, info = env.step(action)
        env.render()

        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1
        if done or steps > env.info.horizon:
            print("J: ", J, " R: ", R)
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()
