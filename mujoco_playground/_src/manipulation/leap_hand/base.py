# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base classes for leap hand."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.leap_hand import leap_hand_constants as consts


def get_assets() -> Dict[str, bytes]:
  assets = {}
  path = mjx_env.MENAGERIE_PATH / "leap_hand"
  mjx_env.update_assets(assets, path / "assets")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(
      assets, consts.ROOT_PATH / "xmls" / "reorientation_cube_textures"
  )
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "meshes")
  return assets


class LeapHandEnv(mjx_env.MjxEnv):
  """Base class for LEAP hand environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=get_assets()
    )
    self._mj_model.opt.timestep = self._config.sim_dt
    # gravity change
    self._mj_model.opt.gravity = self._config.gravity
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model)
    self._xml_path = xml_path

  # Sensor readings.

  def get_palm_position(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "palm_position")

  def get_cube_position(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_position")

  def get_cube_orientation(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_orientation")

  def get_cube_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_linvel")

  def get_cube_angvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_angvel")

  def get_cube_angacc(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_angacc")

  def get_cube_upvector(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_upvector")

  def get_cube_goal_orientation(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_goal_orientation")

  def get_cube_goal_upvector(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "cube_goal_upvector")

  def get_fingertip_positions(self, data: mjx.Data) -> jax.Array:
    """Get fingertip positions relative to the grasp site."""
    return jp.concatenate([
        mjx_env.get_sensor_data(self.mj_model, data, f"{name}_position")
        for name in consts.FINGERTIP_NAMES
    ])
  
  def get_contact_information(self, data: mjx.Data) -> tuple:
    """获取手指尖与 cube 之间的接触信息。
    
    返回一个四元组，依次为：
      - contact_positions: 每个 fingertip 的接触位置（拼接后的 1D 数组）
      - contact_frames: 每个 fingertip 的接触坐标系（扁平化为 1D 数组）
      - contact_forces: 每个 fingertip 的平均接触力
      - contact_torques: 每个 fingertip 的平均接触力矩
    """
    import numpy as np

    hand_tip_names = ['if_tip', 'mf_tip', 'rf_tip', 'th_tip']
    object_name = 'cube'
    positions_list = []
    frames_list = []
    forces_list = []
    torques_list = []

    for tip in hand_tip_names:
        total_pos = np.zeros(3)
        total_frame = np.zeros((3, 3))
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        count = 0

        for i in range(data.ncon):
            contact = data.contact[i]
            geom1_id, geom2_id = contact.geom1, contact.geom2
            body1_id = self.mj_model.geom_bodyid[geom1_id]
            body2_id = self.mj_model.geom_bodyid[geom2_id]
            body1_name = self.mj_model.body(body1_id).name
            body2_name = self.mj_model.body(body2_id).name
            if isinstance(body1_name, bytes):
                body1_name = body1_name.decode()
            if isinstance(body2_name, bytes):
                body2_name = body2_name.decode()
            if ((body1_name == tip and body2_name == object_name) or
                (body2_name == tip and body1_name == object_name)):
                total_pos += contact.pos.copy()
                total_frame += contact.frame.copy().reshape(3, 3)
                force_torque = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.mj_model, data, i, force_torque)
                total_force += force_torque[:3]
                total_torque += force_torque[3:]
                count += 1

        if count > 0:
            avg_pos = total_pos / count
            avg_frame = total_frame / count
            avg_force = total_force / count
            avg_torque = total_torque / count
        else:
            avg_pos = np.zeros(3)
            avg_frame = np.zeros((3, 3))
            avg_force = np.zeros(3)
            avg_torque = np.zeros(3)
        positions_list.append(avg_pos)
        frames_list.append(avg_frame.flatten())
        forces_list.append(avg_force)
        torques_list.append(avg_torque)

    contact_positions = np.concatenate(positions_list)
    contact_frames = np.concatenate(frames_list)
    contact_forces = np.concatenate(forces_list)
    contact_torques = np.concatenate(torques_list)

    # 转换成 jax 数组，保持和其他 sensor 数据一致
    return jp.array(contact_positions), jp.array(contact_frames), jp.array(contact_forces), jp.array(contact_torques)


  # def get_contact_information(self, data:mjx.Data) -> jax.Array:
  #   """Get Contact information between cube and hand"""
  #   object_name = 'cube'
  #   pass

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model


def uniform_quat(rng: jax.Array) -> jax.Array:
  """Generate a random quaternion from a uniform distribution."""
  u, v, w = jax.random.uniform(rng, (3,))
  return jp.array([
      jp.sqrt(1 - u) * jp.sin(2 * jp.pi * v),
      jp.sqrt(1 - u) * jp.cos(2 * jp.pi * v),
      jp.sqrt(u) * jp.sin(2 * jp.pi * w),
      jp.sqrt(u) * jp.cos(2 * jp.pi * w),
  ])
