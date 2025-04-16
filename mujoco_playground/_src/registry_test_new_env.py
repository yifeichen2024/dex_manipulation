# Copyright 2025 DeepMind ...
"""Tests for CubeRotateZAxisFC registration."""

from absl.testing import absltest
from ml_collections import config_dict
from mujoco_playground._src import manipulation
from mujoco_playground._src import registry
from mujoco_playground._src.manipulation.leap_hand import rotate_z_fc

class CubeRotateZAxisFCTest(absltest.TestCase):

  def test_environment_registration(self):
    # 注册环境（如果还没有注册，可手动调用注册函数，或确保在项目初始化时已经注册）
    manipulation.register_environment(
        'CubeRotateZAxisFC',
        rotate_z_fc.CubeRotateZAxisFC,
        rotate_z_fc.default_config,
    )

    # 通过 registry.load 加载环境实例
    env = manipulation.load('CubeRotateZAxisFC')
    self.assertIsNotNone(env)
    self.assertTrue(hasattr(env, 'step'))
    self.assertTrue(hasattr(env, 'reset'))
  
  def test_default_config(self):
    # 获取默认配置
    config = manipulation.get_default_config('CubeRotateZAxisFC')
    # 配置中应包含 force_closure 的权重项
    self.assertIn('force_closure', config.reward_config.scales)
    # 例如，默认值为 1.0
    self.assertEqual(config.reward_config.scales.force_closure, 1.0)

if __name__ == '__main__':
  absltest.main()
