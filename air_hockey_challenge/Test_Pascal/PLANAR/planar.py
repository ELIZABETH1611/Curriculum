import os

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.random
from experiment_launcher import run_experiment, single_experiment
from curriculum_planar import AirHockeyCurriculumPosition 
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_challenge.framework.challenge_core import ChallengeCore
from examples.rl.atacom_agent_wrapper import ATACOMAgent, build_ATACOM_Controller
from examples.rl.network import SACActorNetwork, SACCriticNetwork
from examples.rl.rewards import HitReward, DefendReward, PrepareReward
from examples.rl.rl_agent_wrapper import RlAgent
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Logger, Agent
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length, parse_dataset
from mushroom_rl.utils.frames import LazyFrames
from mushroom_rl.utils.preprocessors import MinMaxPreprocessor
from pathlib import Path


@single_experiment
def experiment(env: str = '3dof-hit',
               alg: str = "sac",
               n_steps: int = 10000,
               n_epochs: int = 500 ,
               quiet: bool = False,
               n_steps_per_fit: int = 1,
               render: bool = True,
               record: bool = False,
               n_eval_episodes: int =100 ,
               slack_type: str = 'soft_corner',
               slack_beta: float = 4,
               slack_tol: float = 1e-6,
               actor_lr: float = 5e-4,
               critic_lr: float = 5e-4,
               n_features: str = "256 256 256 256",
               batch_size: int = 64,
               initial_replay_size: int = 5000,
               max_replay_size: int = 200000,
               tau: float = 0.001,
               warmup_transitions: int = 10000,
               lr_alpha: float = 3e-4,
               use_cuda: bool = False,
               double_integration: bool = False,
               checkpoint: str = "None",
               threshold: float=0.70,
               interpolation_order: int=-1,
               target_entropy: float=-3.0,
               seed: int = 0,
               results_dir: str = './logs',
               **kwargs):
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = Logger(log_name=env, results_dir=results_dir, seed=seed)
    eval_params = {
        "n_episodes": n_eval_episodes,
        "quiet": quiet,
        "render": render,
        "record":record
    }

    kwargs['debug'] = True
    #######################
    ##### Environment######
    random_init      = True    # True: Distance Logarithm for Curriculum  False: RL algorithm
    velocity         = True    # True: Puck with velocity   False: Puck Static
    width_goal       = 0.18    # Value of the width of the target
    mdp = AirHockeyCurriculumPosition(random_init=random_init)
    mdp.env_info['table']['goal_width']= width_goal    # Set the width of the target
    mdp._model.site_size[2] = np.asarray([0.01,mdp.env_info['table']['goal_width']/2,0.001]) # Set in the render the width of the target
    mdp.sparce      = False   # True: Sparce Reward [-10,0,10]
    mdp.moving_init = velocity    # Set velocity 
    mdp.velocity_cl = velocity    # Sets the speed according to curriculum
    mdp.split       = 30      # Number task curriculum 30 for log_Curri 
    mdp.split_2     = 30      # Set task spacing 30 for large steps (logarithm) and 0.1 for small steps (linear)
    mdp.Reduce_sample=mdp.reduce_sample(0.15,0.945,mdp.split)
    value           = 0       # Count the tasks 
    value_2         = value
    mdp.get_distribution(value,value_2)
    #######################

    load = False
    if load:
        results_dir="/home/mcg/REMOTE/curriculum/air_hockey_TU/TEST_9/logs/CRL_3_30_2023-10-11_16-18-09/experiment_id___0/0/3dof-hit/agent-0-best.msh"
        agent_path = Path(results_dir)
        agent = Agent.load(agent_path)
    else:
        agent = agent_builder(mdp.env_info, locals())

    core = ChallengeCore(agent, mdp, action_idx=[0, 1])


    for epoch in range(n_epochs):
        mdp.env_info['table']['goal_width']  =  width_goal
        mdp._model.site_size[2] = np.asarray([0.01,mdp.env_info['table']['goal_width']/2,0.001])
        # Comment learn if you just want to evaluate
        # core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=quiet)
        # Evaluate
        J, R, E, V, alpha, achievement = compute_metrics(core, eval_params)
        rl_agent = core.agent.rl_agent
        logger.log_best_agent(agent, J)
        logger.log_numpy(Success_rate=achievement,J=J,R=R,E=E)
        logger.log_numpy(G_P=mdp.goal, puck=mdp.hit_range) 
        if achievement>=threshold:
            mdp.moving_init=velocity
            mdp.velocity_cl=velocity
            rl_agent._target_entropy=mdp.decay_target_entropy(value)
            logger.log_agent(agent,epoch)
            value+=1
            logger.log_best_agent(agent, J)
            mdp.get_distribution(value,value_2)
            value_2=value
        if (value>=mdp.split) and (value%10==0) :
            logger.log_agent(agent,epoch)

        logger.epoch_info(epoch, J=J, R=R, E=E, V=V)
        
    agent = Agent.load(os.path.join(logger.path, f"agent-{seed}.msh"))

    core = ChallengeCore(agent, mdp, action_idx=[0, 1])

    eval_params["render"] = False
    eval_params["record"] = False

    logger.info('Press a button to visualize')
    J, R, E, V, alpha, achievement = compute_metrics(core, eval_params)



def mdp_builder(env, kwargs):
    settings = {}
    keys = ["gamma", "horizon", "debug", "interpolation_order"]

    for key in keys:
        if key in kwargs.keys():
            settings[key] = kwargs[key]
            del kwargs[key]

    return AirHockeyChallengeWrapper(env, **settings)


def agent_builder(env_info, kwargs):
    alg = kwargs["alg"]

    # If load agent from a checkpoint
    if kwargs["checkpoint"] != "None":
        checkpoint = kwargs["checkpoint"]
        seed = kwargs["seed"]
        del kwargs["checkpoint"]
        del kwargs["seed"]

        for root, dirs, files in os.walk(checkpoint):
            for name in files:
                if name == f"agent-{seed}.msh":
                    agent_dir = os.path.join(root, name)
                    print("Load agent from: ", agent_dir)
                    agent = RlAgent.load(agent_dir)
                    return agent
        raise ValueError(f"Unable to find agent-{seed}.msh in {root}")

    if alg == "sac":
        sac_agent = build_agent_SAC(env_info, **kwargs)
        return RlAgent(env_info, kwargs["double_integration"], sac_agent)

    if alg == "atacom-sac":
        sac_agent = build_agent_SAC(env_info, **kwargs)
        atacom = build_ATACOM_Controller(env_info, **kwargs)
        return ATACOMAgent(env_info, kwargs["double_integration"], sac_agent, atacom)


def build_agent_SAC(env_info, alg, actor_lr, critic_lr, n_features, batch_size,
                    initial_replay_size, max_replay_size, tau,
                    warmup_transitions, lr_alpha, use_cuda,
                    double_integration, **kwargs):
    if type(n_features) is str:
        n_features = list(map(int, n_features.split(" ")))

    from mushroom_rl.utils.spaces import Box
    if double_integration:
        env_info['rl_info'].action_space = Box(*env_info["robot"]["joint_acc_limit"])
    else:
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
                      critic_fit_params=None,
                      )

    agent = SAC(env_info['rl_info'], actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params,
                actor_optimizer=actor_optimizer, critic_params=critic_params, **alg_params)

    return agent


def compute_V(agent, dataset):
    def get_init_states(dataset):
        pick = True
        x_0 = list()
        for d in dataset:
            if pick:
                if isinstance(d[0], LazyFrames):
                    x_0.append(np.array(d[0]))
                else:
                    x_0.append(d[0])
            pick = d[-1]
        return np.array(x_0)

    Q = list()
    for state in get_init_states(dataset):
        s = np.array([state for i in range(100)])
        a = np.array([agent.draw_action(state)[2] for i in range(100)])
        Q.append(agent.rl_agent._critic_approximator(s, a).mean())
    return np.array(Q).mean()


def normalize_state(parsed_state, agent):
    normalized_state = parsed_state.copy()
    for i in range(len(normalized_state)):
        normalized_state[i] = agent.preprocess(normalized_state[i])
    return normalized_state


def compute_metrics(core, eval_params):
    dataset, dataset_info = core.evaluate(**eval_params, get_env_info=True)
    parsed_dataset = parse_dataset(dataset)

    rl_agent = core.agent.rl_agent
    ###############
    # Calculates the success rate
    #############
    if (np.sum(dataset_info['success_task'])+ np.sum(dataset_info['fail_task']))==0:
        achievement=0
    else:
        achievement=np.sum(dataset_info['success_task'])/(np.sum(dataset_info['success_task'])+np.sum(dataset_info['fail_task']))
    ###############
    # End
    #############

    J = np.mean(compute_J(dataset, core.mdp.info.gamma))
    R = np.mean(compute_J(dataset))

    normalized_state = normalize_state(parsed_dataset[0], core.agent)
    _, log_prob_pi = rl_agent.policy.compute_action_and_log_prob(normalized_state)
    # _, log_prob_pi = rl_agent.policy.compute_action_and_log_prob(parsed_dataset[0])
 
    E = -log_prob_pi.mean()

    V = compute_V(core.agent, dataset)

    alpha = rl_agent._alpha

    return J, R, E, V, alpha, achievement


if __name__ == "__main__":
    run_experiment(experiment)
