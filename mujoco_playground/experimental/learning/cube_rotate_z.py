#!/usr/bin/env python3
"""
In-hand Cube Rotation Policy Training Script
==============================================
该脚本整合了 Jupyter Notebook 中的代码，用于在 MuJoCo 环境中基于 Brax 和 PPO 算法训练一个手持立方体旋转任务的控制策略。
请确保安装以下依赖包（例如通过 pip 安装）：
  - jax
  - matplotlib
  - mediapy
  - mujoco
  - wandb
  - brax
  - etils
  - flax
  - orbax
  - mujoco_playground
  - 以及其他依赖库
"""

import os
import functools
import json
import pickle
from datetime import datetime
from pprint import pprint

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import wandb
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
from IPython.display import clear_output, display
from orbax import checkpoint as ocp

from mujoco_playground import manipulation, wrapper
from mujoco_playground.config import manipulation_params
os.environ["JAX_PLATFORM_NAME"] = "cpu"
# ---------------------------
# 1. 环境变量与 JAX 编译缓存设置
# ---------------------------
# 设置 XLA 相关环境变量
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# 设置 JAX 的编译缓存目录
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# ---------------------------
# 2. 全局参数设置
# ---------------------------
env_name = "LeapCubeRotateZAxis"
env_cfg = manipulation.get_default_config(env_name)
randomizer = manipulation.get_domain_randomizer(env_name)
ppo_params = manipulation_params.brax_ppo_config(env_name)
pprint(ppo_params)

# 设置是否使用 wandb 记录日志（如果不需要可保持 False）
USE_WANDB = False
if USE_WANDB:
    wandb.init(project="mjxrl", config=env_cfg)
    wandb.config.update({
        "env_name": env_name,
    })

# 如需在已有检查点上进行 fine-tuning，可设置 FINETUNE_PATH（否则保持 None）
SUFFIX = None
FINETUNE_PATH = None

# ---------------------------
# 3. 实验名称与检查点目录设置
# ---------------------------
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"{env_name}-{timestamp}"
if SUFFIX is not None:
    exp_name += f"-{SUFFIX}"
print(f"Experiment name: {exp_name}")

# 如果设置了 FINETUNE_PATH，则恢复最新的检查点
if FINETUNE_PATH is not None:
    FINETUNE_PATH = epath.Path(FINETUNE_PATH)
    latest_ckpts = list(FINETUNE_PATH.glob("*"))
    latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
    latest_ckpts.sort(key=lambda x: int(x.name))
    latest_ckpt = latest_ckpts[-1]
    restore_checkpoint_path = latest_ckpt
    print(f"Restoring from: {restore_checkpoint_path}")
else:
    restore_checkpoint_path = None

# 创建检查点存储目录，并保存环境配置
ckpt_path = epath.Path("checkpoints").resolve() / exp_name
ckpt_path.mkdir(parents=True, exist_ok=True)
print(f"Checkpoint directory: {ckpt_path}")

with open(ckpt_path / "config.json", "w") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

# ---------------------------
# 4. 定义训练过程中的回调函数
# ---------------------------
# 用于记录训练进度与绘图展示的全局变量
x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

def progress(num_steps, metrics):
    """
    训练进度回调函数
    记录 wandb 日志、更新实时图像展示（当前奖励与标准差）等信息
    """
    if USE_WANDB:
        wandb.log(metrics, step=num_steps)

    # 刷新输出（在 Notebook 中使用 clear_output，在脚本中也会清屏）
    clear_output(wait=True)
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])

    # plt.clf()  # 清除当前图像内容
    # plt.xlim([0, ppo_params.num_timesteps * 1.25])
    # plt.xlabel("# environment steps")
    # plt.ylabel("reward per episode")
    # plt.title(f"Current reward: {y_data[-1]:.3f}")
    # plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
    # # 使用 display 展示图像
    # display(plt.gcf())
    # plt.pause(0.001)  # 暂停以刷新图像

def policy_params_fn(current_step, make_policy, params):
    """
    训练过程中保存模型参数的回调函数
    使用 Orbax 保存检查点
    """
    # 此处参数 make_policy 未使用
    del make_policy
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = ckpt_path / f"{current_step}"
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)

# ---------------------------
# 5. 构建训练函数
# ---------------------------
# 从 ppo_params 中剥离出训练参数并构造训练函数
training_params = dict(ppo_params)
del training_params["network_factory"]

train_fn = functools.partial(
    ppo.train,
    **training_params,
    network_factory=functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    ),
    restore_checkpoint_path=restore_checkpoint_path,
    progress_fn=progress,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    policy_params_fn=policy_params_fn,
    randomization_fn=randomizer,
)

# ---------------------------
# 6. 训练与评估
# ---------------------------
def main():
    # 加载训练环境与评估环境
    env = manipulation.load(env_name, config=env_cfg)
    eval_env = manipulation.load(env_name, config=env_cfg)

    # 调用训练函数，返回推理函数与训练好的参数
    make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env)
    if len(times) > 1:
        print(f"Time for JIT compilation: {times[1] - times[0]}")
        print(f"Training duration: {times[-1] - times[1]}")

    # 绘制奖励随实际运行时间变化图
    plt.figure()
    plt.xlabel("wallclock time (s)")
    plt.ylabel("reward per episode")
    plt.title(f"Final reward: {y_data[-1]:.3f}")
    elapsed_seconds = [(t - times[0]).total_seconds() for t in times[:-1]]
    plt.errorbar(elapsed_seconds, y_data, yerr=y_dataerr, color="blue")
    plt.show()

    # 保存 normalizer, policy 与 value 参数到 pickle 文件
    normalizer_params, policy_params, value_params = params
    with open(ckpt_path / "params.pkl", "wb") as f:
        data = {
            "normalizer_params": normalizer_params,
            "policy_params": policy_params,
            "value_params": value_params,
        }
        pickle.dump(data, f)

    # 生成推理函数（使用确定性策略）
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)

    # 重载评估环境，并对环境的 reset 与 step 方法进行 JIT 编译
    eval_env = manipulation.load(env_name, config=env_cfg)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    rng = jax.random.PRNGKey(123)

    # 初始状态与数据存储列表
    state = jit_reset(rng)
    rollout = [state]
    actions = []
    rewards = []
    cube_angvel = []
    cube_angacc = []

    # Rollout 过程，直到达到 episode_length 或者环境提前结束
    for i in range(env_cfg.episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)
        rewards.append({k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")})
        actions.append({
            "policy_output": ctrl,
            "motor_targets": state.info["motor_targets"],
        })
        cube_angvel.append(env.get_cube_angvel(state.data))
        cube_angacc.append(env.get_cube_angacc(state.data))
        if state.done:
            print("Done detected, stopping rollout.")
            break

    # ---------------------------
    # 7. 分析立方体角速度数据（以 Z 轴角速度为例）
    # ---------------------------
    # 从记录中取出索引 3 的数据作为 Z 轴角速度
    z_angvel = [c[3] for c in cube_angvel]
    z_angvel = jp.array(z_angvel)
    # 通过移动平均平滑数据
    z_angvel = jp.convolve(z_angvel, jp.ones(10) / 10, mode="valid")
    plt.figure()
    plt.plot(z_angvel)
    mean_val = jp.mean(z_angvel)
    print(f"Mean z-axis angular velocity: {mean_val}")
    plt.axhline(mean_val, color="red", linestyle="--", label="mean")
    plt.legend()
    plt.title("Smoothed Z-Axis Angular Velocity")
    plt.xlabel("Time step")
    plt.ylabel("Angular velocity")
    plt.show()

    # ---------------------------
    # 8. 生成训练回放视频
    # ---------------------------
    render_every = 1  # 每一步渲染，可根据需要调整
    fps = 1.0 / eval_env.dt / render_every
    print(f"fps: {fps}")

    traj = rollout[::render_every]

    # 设置 MuJoCo 场景渲染选项
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    # 调整模型的视角与尺寸
    eval_env.mj_model.stat.meansize = 0.02
    eval_env.mj_model.stat.extent = 0.25
    eval_env.mj_model.vis.global_.azimuth = 140
    eval_env.mj_model.vis.global_.elevation = -25

    # 生成渲染帧并展示视频
    frames = eval_env.render(traj, height=480, width=640, scene_option=scene_option)
    media.show_video(frames, fps=fps)
    # 如果需要保存视频文件，可取消下行注释：
    # media.write_video(f"{env_name}.mp4", frames, fps=fps)

if __name__ == "__main__":
    main()
