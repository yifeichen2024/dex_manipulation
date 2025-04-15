# %%
import json
import itertools
import time 
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from datetime import datetime
import functools
import os
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp
import mujoco.viewer
# <key
#   time="0"
#   qpos="0.74176 0.14658 0.486265 0.74168 0.65272 0 0.677545 0.58516 0.46192 -0.27222 1.15574 0.28416 1.81305 0 0.214555 0.431 0.15 0 0.05 0.810967 -0.00262895 -0.585086 -0.000254303"
#   qvel="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
#   ctrl="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
#   mpos="0.325 0.17 0.0475"
#   mquat="1 0 0 0"
# />
# %%
#@title Import The Playground

from mujoco_playground import wrapper
from mujoco_playground import registry
# %% 
class MyEnvironmentClass:
    def __init__(self, config):
        self.config = config        

    def reset(self):        
        pass

    def step(self, action):        
        pass

default_config_of_my_environment = {
    'param1': 'value1',
    'param2': 'value2',    
}

# Register the environment
# This directlty give back the env to 
registry.manipulation.register_environment("NewEnv", MyEnvironmentClass, default_config_of_my_environment)

# %%
registry.manipulation.ALL_ENVS

# %%
env_name = 'LeapCubeRotateZAxis'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)
env_cfg
# %% 
xml_path = "mujoco_playground/_src/manipulation/leap_hand/xmls/scene_mjx_cube_flip.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "grasp")
mujoco.mj_resetDataKeyframe(model, data, kf_id)
print(model.opt.timestep)
viewer = mujoco.viewer.launch(model, data)
# # %%
# from mujoco_playground.config import manipulation_params
# ppo_params = manipulation_params.brax_ppo_config(env_name)
# ppo_params
# # %%
# x_data, y_data, y_dataerr = [], [], []
# times = [datetime.now()]


# def progress(num_steps, metrics):
#   clear_output(wait=True)

#   times.append(datetime.now())
#   x_data.append(num_steps)
#   y_data.append(metrics["eval/episode_reward"])
#   y_dataerr.append(metrics["eval/episode_reward_std"])

#   plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
#   plt.xlabel("# environment steps")
#   plt.ylabel("reward per episode")
#   plt.title(f"y={y_data[-1]:.3f}")
#   plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

#   plt.show(plt.gcf())

# ppo_training_params = dict(ppo_params)
# network_factory = ppo_networks.make_ppo_networks
# if "network_factory" in ppo_params:
#   del ppo_training_params["network_factory"]
#   network_factory = functools.partial(
#       ppo_networks.make_ppo_networks,
#       **ppo_params.network_factory
#   )

# train_fn = functools.partial(
#     ppo.train, **dict(ppo_training_params),
#     network_factory=network_factory,
#     progress_fn=progress,
#     seed=1
# )
# # %%
# make_inference_fn, params, metrics = train_fn(
#     environment=env,
#     wrap_env_fn=wrapper.wrap_for_brax_training,
# )
# print(f"time to jit: {times[1] - times[0]}")
# print(f"time to train: {times[-1] - times[1]}")
# # %%
# jit_reset = jax.jit(env.reset)
# jit_step = jax.jit(env.step)
# jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

# #%%
# rng = jax.random.PRNGKey(42)
# rollout = []
# n_episodes = 1

# for _ in range(n_episodes):
#   state = jit_reset(rng)
#   rollout.append(state)
#   for i in range(env_cfg.episode_length):
#     act_rng, rng = jax.random.split(rng)
#     ctrl, _ = jit_inference_fn(state.obs, act_rng)
#     state = jit_step(state, ctrl)
#     rollout.append(state)

# render_every = 1
# frames = env.render(rollout[::render_every])
# rewards = [s.reward for s in rollout]
# media.show_video(frames, fps=1.0 / env.dt / render_every)