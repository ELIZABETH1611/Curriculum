import torch
import argparse
import numpy as np
from pathlib import Path
import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.core import Core
from mushroom_rl.algorithms.actor_critic import SAC
from examples.rl.atacom_agent_wrapper import ATACOMAgent
from examples.rl.atacom.system import VelocityControlSystem
from examples.rl.atacom import ATACOMController, ConstraintList
from examples.rl.network import SACActorNetwork, SACCriticNetwork
from experiments.currot.currot import GPUCurrOT, ContextTransform
from experiments.currot.gpu_currot_utils import ParameterSchedule
from experiments.environments.boundary_distributions import get_virtual_boundaries
from experiments.environments.air_hockey_wrapper import AirHockeyWrapper, AirHockeyCurriculum
from examples.rl.air_hockey_contraints import JointPosConstraint, EndEffectorPosConstraint


class CurrOTWrapper(AirHockeyCurriculum):

    def __init__(self, currot: GPUCurrOT):
        self.currot = currot

    def __call__(self, n: int = 1, concatenated: bool = False):
        contexts = self.currot.sample(torch.arange(n)).numpy()
        if concatenated:
            return np.squeeze(contexts)
        else:
            return np.squeeze(contexts[..., :2]), np.squeeze(contexts[..., 2:4]), np.squeeze(contexts[..., 4:])

    def update(self, contexts: np.ndarray, successes: np.ndarray):
        self.currot.update_distribution(torch.from_numpy(contexts).float(),
                                        torch.from_numpy(successes).float()[:, None])

    def save(self, path):
        self.currot.save(path)


class AirHockeyContextTransform(ContextTransform):

    def __init__(self, velocity_bounds=None):
        super().__init__()
        table_bounds = ([-0.974, -0.519], [0.974, 0.519])
        if velocity_bounds is None:
            velocity_bounds = ([-0.2, -0.2], [0., 0.2])
        self.lb = torch.tensor(table_bounds[0] + table_bounds[0] + velocity_bounds[0])
        self.ub = torch.tensor(table_bounds[1] + table_bounds[1] + velocity_bounds[1])

    @staticmethod
    def _colored_constraint_fn(context):
        goal_pos = context[..., :2]
        puck_pos = context[..., 2:4]
        puck_vel = context[..., 4:]

        bounds = (torch.tensor([-0.75, -0.519]), torch.tensor([0.974, 0.519]))
        vel_bounds = (torch.tensor([-1, -0.2]), torch.tensor([0, 0.2]))

        goal_in_bounds = torch.logical_and(torch.all(bounds[0] <= goal_pos, dim=-1),
                                           torch.all(goal_pos <= bounds[1], dim=-1))
        puck_in_bounds = torch.logical_and(torch.all(bounds[0] <= puck_pos, dim=-1),
                                           torch.all(puck_pos <= bounds[1], dim=-1))
        vel_in_bounds = torch.logical_and(torch.all(vel_bounds[0] <= puck_vel, dim=-1),
                                          torch.all(puck_vel <= vel_bounds[1], dim=-1))
        in_bounds = torch.logical_and(goal_in_bounds, np.logical_and(puck_in_bounds, vel_in_bounds))

        # The puck pos needs to be at least 5 cm away from the goal position in each direction
        lb, ub = get_virtual_boundaries(goal_pos.numpy(), 1.948)
        lb, ub = torch.from_numpy(lb).float(), torch.from_numpy(ub).float()
        goal_distance_ok = torch.logical_and(
            torch.logical_and(puck_pos[..., 0] >= lb[..., 0] + 0.05, puck_pos[..., 0] <= ub[..., 0] - 0.05),
            torch.logical_and(puck_pos[..., 1] >= lb[..., 1] + 0.05, puck_pos[..., 1] <= ub[..., 1] - 0.05))

        return torch.logical_and(goal_distance_ok, in_bounds)

    def constraint_fn(self, x: torch.Tensor):
        return self._colored_constraint_fn(self.color(x))

    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.lb) / (self.ub - self.lb)

    def color(self, x: torch.Tensor) -> torch.Tensor:
        return self.lb + x * (self.ub - self.lb)


def build_ATACOM_Controller(env_info, slack_type, slack_beta, slack_tol):
    lambda_c = 1 / env_info['dt']
    dim_q = env_info['robot']['n_joints']

    constraint_list = ConstraintList(dim_q)
    constraint_list.add_constraint(JointPosConstraint(env_info))
    constraint_list.add_constraint(EndEffectorPosConstraint(env_info))
    system = VelocityControlSystem(dim_q, env_info['robot']['joint_vel_limit'][1] * 0.95)

    return ATACOMController(constraint_list, system, slack_beta=slack_beta,
                            slack_dynamics_type=slack_type, slack_tol=slack_tol, lambda_c=lambda_c)


def build_agent_SAC(env_info, alg, actor_lr, critic_lr, n_features, batch_size, initial_replay_size, max_replay_size,
                    tau, warmup_transitions, lr_alpha, use_cuda):
    if type(n_features) is str:
        n_features = list(map(int, n_features.split(" ")))

    from mushroom_rl.utils.spaces import Box
    env_info['rl_info'].action_space = Box(*env_info["robot"]["joint_vel_limit"])

    if alg == "atacom-sac":
        env_info['rl_info'].action_space = Box(-np.ones(env_info['robot']['n_joints']),
                                               np.ones(env_info['robot']['n_joints']))
        obs_low = np.concatenate([env_info['rl_info'].observation_space.low, -np.ones(env_info['robot']['n_joints'])])
        obs_high = np.concatenate([env_info['rl_info'].observation_space.high, np.ones(env_info['robot']['n_joints'])])
        env_info['rl_info'].observation_space = Box(obs_low, obs_high)

    actor_mu_params = dict(network=SACActorNetwork,
                           input_shape=env_info["rl_info"].observation_space.shape,
                           output_shape=env_info["rl_info"].action_space.shape,
                           n_features=n_features,
                           use_cuda=use_cuda)
    actor_sigma_params = dict(network=SACActorNetwork,
                              input_shape=env_info["rl_info"].observation_space.shape,
                              output_shape=env_info["rl_info"].action_space.shape,
                              n_features=n_features,
                              use_cuda=use_cuda)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': actor_lr}}
    critic_params = dict(network=SACCriticNetwork,
                         input_shape=(env_info["rl_info"].observation_space.shape[0] +
                                      env_info["rl_info"].action_space.shape[0],),
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': critic_lr}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    alg_params = dict(initial_replay_size=initial_replay_size,
                      max_replay_size=max_replay_size,
                      batch_size=batch_size,
                      warmup_transitions=warmup_transitions,
                      tau=tau,
                      lr_alpha=lr_alpha,
                      critic_fit_params=None)

    agent = SAC(env_info['rl_info'], actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params,
                actor_optimizer=actor_optimizer, critic_params=critic_params, **alg_params)

    return agent


def evaluate_agent(eval_core: Core, save_path: Path, render: bool = False):
    dataset, info = eval_core.evaluate(n_episodes=100, get_env_info=True, render=render)
    successes = []
    interceptions = []
    for i in range(len(dataset)):
        if dataset[i][-1]:
            successes.append(info["success_task"][i])
            interceptions.append(info["intercepted"][i])
    np.savez(save_path, successes=np.array(successes), interceptions=np.array(interceptions))


def main(robot: str, use_atacom: bool, curriculum: bool, n_steps_per_fit: int, curriculum_epsilon: float = 0.2,
         min_success_rate: float = 0.5, render: bool = False, save_path: Path = None, seed: int = 0,
         action_penalty: float = 0., sparse_reward: bool = False, n_evaluations: int = 0, n_checkpoints: int = 5):
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    from experiments.environments.boundary_distributions import TargetBoundarySampler, EasyBoundarySampler
    if curriculum == "hand_crafted":
        from experiments.environments.boundary_distributions import BoundaryCurriculum
        task_sampler = BoundaryCurriculum(20, performance_threshold=min_success_rate)
    elif curriculum == "currot":
        context_transform = AirHockeyContextTransform()

        easy_samples = EasyBoundarySampler()(1000, concatenated=True)
        # We initially shoot the pucks towards the agent
        easy_samples[:, 4:] = 5 * (np.array([-0.75, 0.])[None, :] - easy_samples[:, 2:4])
        assert torch.all(
            context_transform.constraint_fn(context_transform.whiten(torch.from_numpy(easy_samples).float())))

        target_sampler = TargetBoundarySampler()
        task_sampler = CurrOTWrapper(GPUCurrOT(
            init_samples=torch.from_numpy(easy_samples).float(),
            target_sampler=lambda n: torch.from_numpy(target_sampler(n, concatenated=True)).float(),
            metric_transform=lambda x: x[..., 0] - min_success_rate,
            epsilon=ParameterSchedule(torch.tensor(curriculum_epsilon)),
            constraint_fn=context_transform.constraint_fn, theta=0.25 * torch.pi, success_buffer_size=1000,
            buffer_size=1500, context_transform=context_transform))
    elif curriculum == "none":
        task_sampler = TargetBoundarySampler()
    else:
        raise RuntimeError(f"Unknown curriculum type: '{curriculum}'")

    if use_atacom:
        if robot == "iiwa":
            from experiments.environments.iiwa import AirHockeyPosition
        else:
            from experiments.environments.planar3dof import AirHockeyPosition

        env = AirHockeyWrapper(AirHockeyPosition, curriculum=task_sampler, action_penalty_scale=action_penalty,
                               sparse=sparse_reward)
        eval_env = AirHockeyWrapper(AirHockeyPosition, curriculum=TargetBoundarySampler(),
                                    action_penalty_scale=action_penalty, sparse=sparse_reward)
        sac_agent = build_agent_SAC(env.env_info, alg="atacom-sac", actor_lr=5e-4, critic_lr=5e-4,
                                    n_features="256 256 256 256", batch_size=64, initial_replay_size=2000,
                                    max_replay_size=200000, tau=0.001, warmup_transitions=4000, lr_alpha=3e-4,
                                    use_cuda=False)
        atacom = build_ATACOM_Controller(env.env_info, slack_type='soft_corner', slack_beta=4, slack_tol=1e-6)
        agent = ATACOMAgent(env.env_info, double_integration=False, rl_agent=sac_agent, atacom_controller=atacom)
    else:
        if robot == "iiwa":
            from experiments.environments.iiwa import AirHockey
        else:
            from experiments.environments.planar3dof import AirHockey

        env = AirHockeyWrapper(AirHockey, curriculum=task_sampler, action_penalty_scale=action_penalty,
                               sparse=sparse_reward)
        eval_env = AirHockeyWrapper(AirHockey, curriculum=TargetBoundarySampler(), action_penalty_scale=action_penalty,
                                    sparse=sparse_reward)
        agent = build_agent_SAC(env.env_info, alg="sac", actor_lr=5e-4, critic_lr=5e-4,
                                n_features="256 256 256 256", batch_size=64, initial_replay_size=2000,
                                max_replay_size=200000, tau=0.001, warmup_transitions=4000, lr_alpha=3e-4,
                                use_cuda=False)

    core = Core(agent, env)
    eval_core = Core(agent, eval_env)

    save_epochs = np.round(np.linspace(0, 49, n_checkpoints)[1:]).astype(np.int64)
    eval_epochs = np.round(np.linspace(0, 49, n_evaluations)[1:]).astype(np.int64)
    if save_path is not None and n_checkpoints > 0:
        agent.save(save_path / "agent_0")

    if save_path is not None and n_evaluations > 0:
        evaluate_agent(eval_core, save_path / "agent_performance_0.npz")

    for i in range(50):
        print(f"Epoch: {i}")
        core.learn(n_steps=100000, n_steps_per_fit=n_steps_per_fit, quiet=False, render=render)
        # Save the data
        if save_path is not None and i == save_epochs[0]:
            agent.save(save_path / f"agent_{i + 1}")

        # The curriculum checkpoints do not take a lot of space but are useful in debugging
        if save_path is not None:
            if curriculum != "none":
                curriculum_save_path = save_path / f"curriculum_{i + 1}"
                curriculum_save_path.mkdir(exist_ok=True, parents=True)
                task_sampler.save(curriculum_save_path)
            save_epochs = save_epochs[1:]

        if save_path is not None and i == eval_epochs[0]:
            evaluate_agent(eval_core, save_path / f"agent_performance_{i}.npz")
            eval_epochs = eval_epochs[1:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["iiwa", "planar"], type=str, default="iiwa")
    parser.add_argument("--use_atacom", action="store_true")
    parser.add_argument("--sparse_reward", action="store_true")
    parser.add_argument("--action_penalty", type=float, default=0.)
    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument("--curriculum", choices=["hand_crafted", "currot", "none"], type=str, default="none")
    parser.add_argument("--curriculum_success_rate", type=float)

    parser.add_argument("--n_steps_per_fit", type=int)
    parser.add_argument("--n_evaluations", type=int, default=0)
    parser.add_argument("--n_checkpoints", type=int, default=5)

    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    parent_dir = f"{'atacom_' if args.use_atacom else ''}{'iiwa' if args.robot == 'iiwa' else 'planar'}" \
                 f"{'_sparse' if args.sparse_reward else '_dense'}_ap_{args.action_penalty}"
    parent_dir = Path(__file__).resolve().parent / parent_dir

    if args.curriculum == "currot":
        if args.curriculum_success_rate is None:
            raise RuntimeError("If using a curriculum, a success rate needs to be specified!")

        save_dir = parent_dir / "currot_learner" / f"seed_{args.seed}_delta_{args.curriculum_success_rate}"
    elif args.curriculum == "hand_crafted":
        if args.curriculum_success_rate is None:
            raise RuntimeError("If using a curriculum, a success rate needs to be specified!")

        save_dir = parent_dir / "curriculum_learner" / f"seed_{args.seed}_delta_{args.curriculum_success_rate}"
    else:
        save_dir = parent_dir / "default_learner" / f"seed_{args.seed}"
    save_dir.mkdir(exist_ok=True, parents=True)

    main(robot=args.robot, use_atacom=args.use_atacom, curriculum=args.curriculum, curriculum_epsilon=0.2,
         min_success_rate=args.curriculum_success_rate, render=args.render, save_path=save_dir, seed=args.seed,
         action_penalty=args.action_penalty, sparse_reward=args.sparse_reward, n_evaluations=args.n_evaluations,
         n_checkpoints=args.n_checkpoints, n_steps_per_fit=args.n_steps_per_fit)
