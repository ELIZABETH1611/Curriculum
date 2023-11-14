import numpy as np
from abc import ABC, abstractmethod
from mushroom_rl.core.environment import Environment


class AirHockeyCurriculum(ABC):

    @abstractmethod
    def __call__(self, n: int = 1, concatenated: bool = False):
        pass

    @abstractmethod
    def update(self, contexts: np.ndarray, successes: np.ndarray):
        pass


class AirHockeyWrapper(Environment):

    def __init__(self, env_cls, curriculum: AirHockeyCurriculum, *args, action_penalty_scale: float = 0.,
                 episodes_per_update: int = 50, **kwargs):
        self.env = env_cls(task_sampler=self.curriculum_wrapper, *args, **kwargs)
        self.env_info = self.env.env_info
        super().__init__(self.env.info)

        self.step_count = 0
        self.context_buffer = []
        self.success_buffer = []
        self.current_context = None
        self.curriculum = curriculum
        self.action_penalty_scale = action_penalty_scale
        self.episodes_per_update = episodes_per_update

    def curriculum_wrapper(self):
        context = self.curriculum(n=1, concatenated=True)
        self.current_context = np.copy(context)
        return context[:2], context[2:4], context[4:]

    def reset(self, state=None):
        self.step_count = 0
        return self.env.reset(state)

    def step(self, action):
        self.step_count += 1
        if len(action.shape) == 1:
            next_state, reward, done, info = self.env.step(action)
            action_penalty = self.action_penalty_scale * np.sum(np.square(action))
        else:
            next_state, reward, done, info = self.env.step(action[:2])
            action_penalty = self.action_penalty_scale * np.sum(np.square(action[-1, :]))

        # If there is a reset, update the curriculum
        if done or self.step_count >= self.info.horizon:
            # Check if there was a success or failure
            self.context_buffer.append(self.current_context.copy())
            self.success_buffer.append([info["success_task"][0], info["intercepted"][0]])

        if len(self.context_buffer) >= 50:
            suc_buf = np.array(self.success_buffer)
            print(f"Updating Curriculum. Success Rate: {np.mean(suc_buf[:, 0])}, "
                  f"Interception Rate: {np.mean(suc_buf[:, 1])}")
            self.curriculum.update(np.stack(self.context_buffer, axis=0), suc_buf[:, 0].astype(np.float64))

            self.context_buffer = []
            self.success_buffer = []

        return next_state, reward - action_penalty, done, info

    def render(self, record=False):
        self.env.render(record=record)

    def stop(self):
        self.env.stop()
