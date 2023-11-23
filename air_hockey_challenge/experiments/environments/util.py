import math
import mujoco
import numpy as np
from typing import Dict
from mushroom_rl.utils.spaces import Box
from experiments.environments.boundary_distributions import get_virtual_boundaries
from air_hockey_challenge.utils.kinematics import forward_kinematics


def get_ee_pos(env, obs):
    q_pos, q_vel = env.get_joints(obs)
    x_pos, rot_mat = forward_kinematics(env.env_info['robot']['robot_model'],
                                        env.env_info['robot']['robot_data'], q_pos)
    return x_pos + env.env_info['robot']['base_frame'][0][:3, 3]


def get_puck_angle(env, obs):
    puck_pos, puck_vel = env.get_puck(obs)
    ee_pos = get_ee_pos(env, obs)
    puck_theta = math.atan((puck_pos[0] - ee_pos[0]) / (puck_pos[1] - ee_pos[1]))
    if 0.0 > puck_theta >= -np.pi / 2:
        puck_theta = puck_theta + np.pi
    else:
        puck_theta = puck_theta

    return puck_theta


def check_border_contact(env, prev_obs, cur_obs, table_bounds, dt, eps=1e-3):
    table_lb, table_ub = table_bounds
    puck_radius = env.env_info["puck"]["radius"]
    table_lb = table_lb + puck_radius + eps
    table_ub = table_ub - puck_radius - eps

    prev_puck_pos, prev_puck_vel = env.get_puck(prev_obs)
    prev_puck_pos = prev_puck_pos[:2]
    prev_puck_vel = prev_puck_vel[:2]
    puck_pos, puck_vel = env.get_puck(cur_obs)
    puck_pos = puck_pos[:2]
    puck_vel = puck_vel[:2]

    # The only thing we need to be careful with are collisions - but we can detect them by checking whether the
    # orientation of the velocity changed significantly
    collision = False
    if np.linalg.norm(prev_puck_vel) >= 5e-3 and np.linalg.norm(puck_vel) >= 5e-3:
        collision = np.abs(np.linalg.norm(prev_puck_vel) - np.linalg.norm(puck_vel)) > 2e-2 or \
                    np.dot(prev_puck_vel / (np.linalg.norm(prev_puck_vel) + 1e-8),
                           puck_vel / (np.linalg.norm(puck_vel) + 1e-8)) < 0.9

    # If there was a collision, we integrate the previous position to check for a boundary collision if the current step
    # does not yield a collision
    # Obviously this makes only sense if the puck had some velocity
    out_of_bounds = np.any(puck_pos <= table_lb) or np.any(puck_pos >= table_ub)
    if collision and not out_of_bounds:
        puck_pos = prev_puck_pos + dt * prev_puck_vel
        out_of_bounds = np.any(puck_pos <= table_lb) or np.any(puck_pos >= table_ub)

    if out_of_bounds:
        # We compute a more accurate collision point if we hit the table boundaries
        dpos = puck_pos - prev_puck_pos
        intercept_times = np.concatenate(((table_lb - prev_puck_pos) / dpos, (table_ub - prev_puck_pos) / dpos), axis=0)
        collided = np.logical_and(intercept_times >= 0, intercept_times <= 1)
        intercept_times = np.where(collided, intercept_times, np.inf)

        return intercept_times, prev_puck_pos[None, :] + intercept_times[:, None] * dpos[None, :]
    else:
        return np.ones(4) * np.inf, np.ones((4, 2)) * np.inf


def all_absorbing_fn(env, obs, prev_puck_intercepted: bool):
    '''
    Absrobing states
    '''

    table_boundary = np.array([env.env_info['table']['length'], env.env_info['table']['width']]) / 2
    ee_pos = get_ee_pos(env, obs)
    mallet_collision = np.any(np.abs(np.asarray(ee_pos[:2])) > table_boundary)
    puck_pos, puck_vel = env.get_puck(obs)
    velocity_exceeded = np.linalg.norm(puck_vel[:2]) > 100

    goal_boundary = get_virtual_boundaries(env.goal, env.env_info['table']['length'])
    collision_time, collision_pos = check_border_contact(env, env._obs, obs, goal_boundary,
                                                         env._timestep * env._n_intermediate_steps)
    collision = ~np.isinf(collision_time)
    first_collision = np.argmin(collision_time)
    back_collision = np.any(collision) and first_collision == 0

    # In this case, any collision terminates the experiment
    internal_info = {"done": False, "penalty": False, "distance": np.linalg.norm(puck_pos[:2] - env.goal),
                     "success": False, "puck_intercepted": puck_vel[0] >= 0.05 or prev_puck_intercepted}
    # first_collision == 0 means that the puck hit the back end of the table
    if mallet_collision or velocity_exceeded or back_collision:
        internal_info["done"] = True
        internal_info["penalty"] = True
    elif np.any(collision):
        internal_info["done"] = True
        puck_pos = collision_pos[np.argmin(collision_time)]
        internal_info["distance"] = np.linalg.norm(puck_pos - env.goal)
        threshold = env.env_info['table']['goal_width'] / 2.0
        internal_info["success"] = internal_info["distance"] < threshold

    return internal_info


def top_absorbing_fn(env, obs, prev_puck_intercepted: bool):
    '''
    Absorbing states (ignores the sides of the table)
    '''

    table_boundary = np.array([env.env_info['table']['length'], env.env_info['table']['width']]) / 2
    ee_pos = get_ee_pos(env, obs)
    mallet_collision = np.any(np.abs(np.asarray(ee_pos[:2])) > table_boundary)
    puck_pos, puck_vel = env.get_puck(obs)
    velocity_exceeded = np.linalg.norm(puck_vel[:2]) > 100
    puck_out = np.any(np.abs(puck_pos[:2]) > table_boundary)

    # We cap the boundary for the upper end of the table to the goal position
    goal_boundary = (np.array([-env.env_info['table']['length'] / 2, -env.env_info['table']['width'] / 2]),
                     np.array([env.goal[0], env.env_info['table']['width'] / 2]))
    collision_time, collision_pos = check_border_contact(env, env._obs, obs, goal_boundary,
                                                         env._timestep * env._n_intermediate_steps)
    collision = ~np.isinf(collision_time)
    first_collision = np.argmin(collision_time)
    back_collision = np.any(collision) and first_collision == 0
    top_collision = collision[2]

    # In this case, any collision terminates the experiment
    internal_info = {"done": False, "penalty": False, "distance": np.linalg.norm(puck_pos[:2] - env.goal),
                     "success": False, "puck_intercepted": puck_vel[0] >= 0.05 or prev_puck_intercepted}
    # first_collision == 0 means that the puck hit the back end of the table
    if mallet_collision or velocity_exceeded or back_collision or puck_out:
        internal_info["done"] = True
        internal_info["penalty"] = True
    elif top_collision:
        internal_info["done"] = True
        puck_pos = collision_pos[np.argmin(collision_time)]
        internal_info["distance"] = np.linalg.norm(puck_pos - env.goal)
        threshold = env.env_info['table']['goal_width'] / 2.0
        internal_info["success"] = internal_info["distance"] < threshold

    return internal_info


def reward_fn(internal_info: Dict, sparse: bool, gamma: float):
    '''
            Reward
            Sparce:
                    True    - Activate Sparce reward [-10,0,10]
                    False   - Activate Dense Reward
            '''

    puck_pos_initial = np.array([-0.65, -0.35])
    final_goal_pos = np.array([0.974, 0])
    scale = np.linalg.norm(puck_pos_initial - final_goal_pos)

    if internal_info["done"]:
        if internal_info["penalty"]:
            r = -10. if sparse else -60.
        else:
            if internal_info["success"]:
                r = 10. if sparse else 0.
            else:
                r = 0. if sparse else -internal_info["distance"] / (scale * (1 - gamma))
    else:
        r = 0. if sparse else -internal_info["distance"] / scale

    return r


def setup_fn(cls, env, state=None):
    '''
            Add the extra infomation
            Args:
                Goal: Position of the goal
                Theta: Angle between the mallet and the puck
            Returns:
                All the mdp_info
            '''

    env.goal, puck_pos, puck_vel = env.task_sampler()
    # Only for testing
    # puck_vel = env.goal - puck_pos
    env._model.site_pos[mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, "target")] = np.concatenate(
        (env.goal, 0.0), axis=None)
    boundaries = get_virtual_boundaries(env.goal, env.env_info['table']['length'])
    env._model.site_pos[mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, "left_boundary"), 1] = boundaries[0][1]
    env._model.site_pos[mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, "top_boundary"), 0] = boundaries[1][0]
    env._model.site_pos[mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, "right_boundary"), 1] = boundaries[1][1]

    env._write_data("puck_x_vel", puck_vel[0])
    env._write_data("puck_y_vel", puck_vel[1])
    env._write_data("puck_yaw_vel", 0.)

    env._write_data("puck_x_pos", puck_pos[0])
    env._write_data("puck_y_pos", puck_pos[1])

    env._internal_info = None

    super(cls, env).setup(state)


def modify_info_fn(cls, env, mdp_info):
    '''
            Same as the original
            Args:
                Goal:  Min and max Position of the goal
                Theta:  Min and max between the mallet and the puck
            Returns:
                All the mdp_info
            '''
    mdp_info = super(cls, env)._modify_mdp_info(mdp_info)
    obs_low = np.concatenate([mdp_info.observation_space.low, [-0.6, -0.4, -1, -1]])
    obs_high = np.concatenate([mdp_info.observation_space.high, [1., 0.4, 1, 1]])
    mdp_info.observation_space = Box(obs_low, obs_high)
    return mdp_info


def modify_obs_fn(cls, env, obs):
    '''
    Add the extra infomation
    Args:
        Goal: Position of the goal
        Theta: Angle between the mallet and the puck
    Returns:
        All the mdp_info
    '''

    puck_theta = get_puck_angle(env, obs)
    obs = super(cls, env)._modify_observation(obs)
    obs = np.concatenate([obs, [env.goal[0], env.goal[1], math.sin(puck_theta), math.cos(puck_theta)]])
    return obs
