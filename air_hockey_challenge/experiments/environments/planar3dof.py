import numpy as np
from experiments.environments import util
from air_hockey_challenge.environments.planar.single import AirHockeySingle
from air_hockey_challenge.environments.position_control_wrapper import PositionControlPlanar


class AirHockey(AirHockeySingle):
    """
    Class for the air hockey hitting task using CL.
    """

    def __init__(self, task_sampler, gamma=0.99, horizon=500, viewer_params={}, sparse: bool = False):
        """
        Constructor
        Args:
            moving_init(bool, False): If true, initialize the puck with inital velocity.
        """
        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

        self.sparse = sparse
        self.gamma = gamma

        self._internal_info = None

        self.goal = np.array([0.974, 0.519])
        self.task_sampler = task_sampler

    def _modify_mdp_info(self, mdp_info):
        return util.modify_info_fn(AirHockey, self, mdp_info)

    def _modify_observation(self, obs):
        return util.modify_obs_fn(AirHockey, self, obs)

    def setup(self, state=None):
        util.setup_fn(AirHockey, self, state=state)

    def reward(self, state, action, next_state, absorbing):
        return util.reward_fn(self._internal_info, sparse=self.sparse, gamma=self.info.gamma)

    def is_absorbing(self, obs):
        self._internal_info = util.all_absorbing_fn(self, obs)
        # self._internal_info = util.top_absorbing_fn(self, obs)
        return self._internal_info["done"]

    def _create_info_dictionary(self, obs):
        '''
        Create dictionary to save when the task was accomplished or not
        '''

        return {"success_task": np.array([self._internal_info["success"]]),
                "intercepted": np.array([self._internal_info["puck_intercepted"]])}


class AirHockeyPosition(PositionControlPlanar, AirHockey):
    pass


if __name__ == '__main__':
    from experiments.environments.boundary_distributions import TargetBoundarySampler, BoundaryCurriculum
    import matplotlib.pyplot as plt

    task_sampler = BoundaryCurriculum(n_steps=10, performance_threshold=0.)

    for i in range(10):
        goal_pos, puck_pos, puck_vel = task_sampler(n=1000)
        assert np.all(puck_vel == 0)
        plt.scatter(goal_pos[:, 0], goal_pos[:, 1], color="C0")
        plt.scatter(puck_pos[:, 0], puck_pos[:, 1], color="C1")
        plt.xlim([-0.7, 1.0])
        plt.ylim([-0.6, 0.6])
        plt.show()
        task_sampler.update(None, np.ones(10))

    task_sampler = BoundaryCurriculum(n_steps=10, performance_threshold=0.)
    env = AirHockey(task_sampler, sparse=True)
    # env = AirHockeyCurriculumPosition(task_sampler=task_sampler, sparse=True)

    env.reset()
    env.render()
    R = 0.
    J = 0.
    performances = []
    gamma = 1.
    steps = 0
    reset_count = 0
    while True:
        action = np.zeros(3)  # np.random.uniform(-1, 1, (2, 3))  # np.zeros(3)

        observation, reward, done, info = env.step(action)
        env.render()

        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1
        if done or steps > env.info.horizon:
            performances.append(R > 0.)

            print("J: ", J, " R: ", R)
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            reset_count += 1

            if len(performances) > 5:
                print("Stepping Curriculum")
                task_sampler.update(None, np.array(performances).astype(np.float64))
                performances = []

            env.reset()
