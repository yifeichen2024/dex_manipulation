# %%
import time
import copy
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt 
# import dex_manipualtion_env 
xml_path = "./mujoco_playground/_src/manipulation/leap_hand/xmls/scene_mjx_cube.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, kf_id)
print(model.opt.timestep)
viewer = mujoco.viewer.launch(model, data)

# kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "table grasp")
# mujoco.mj_resetDataKeyframe(model, data, kf_id)
# %%
# === JOINT POSISTION ===
# Force torque sensor
# joint_sensor_values = {}

# for i in range(model.nsensor):
#     name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
#     stype = model.sensor_type[i]

#     if stype == mujoco.mjtSensor.mjSENS_JOINTPOS:
#         start = model.sensor_adr[i]
#         dim = model.sensor_dim[i]
#         joint_sensor_values[name] = data.sensordata[start : start + dim].copy()

# for name, values in joint_sensor_values.items():
#     print(f"{name}: {values}")
# # 合并成一个一维向量
# joint_sensor_vector = np.concatenate(list(joint_sensor_values.values()))
# print("Combined joint position sensor:", "Shape: ", len(joint_sensor_vector), "Values: ", joint_sensor_vector)

# joint_ctrl_values = data.ctrl[0:16]
# print("Joint control: ", "Shape: ", len(joint_ctrl_values), "Values: ", joint_ctrl_values)
# print("Joint error: ", joint_ctrl_values - joint_sensor_vector)
# # %%
# # === FORCE TORQUE SENSOR ===
# # Force torque sensor
# FT_sensor_values = {}

# for i in range(model.nsensor):
#     name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
#     stype = model.sensor_type[i]

#     if stype in (mujoco.mjtSensor.mjSENS_FORCE, mujoco.mjtSensor.mjSENS_TORQUE):
#         start = model.sensor_adr[i]
#         dim = model.sensor_dim[i]
#         FT_sensor_values[name] = data.sensordata[start : start + dim].copy()

# for name, values in FT_sensor_values.items():
#     print(f"{name}: {values}")
# # 合并成一个一维向量
# FT_sensor_vector = np.concatenate(list(FT_sensor_values.values()))
# print("Combined Force/torque sensor:", FT_sensor_vector)

# # %%
# # Cube information
# # Observation: cube_position, cube_orientation, cube_angvel, cube_upvector
# sensor_values = {}

# for i in range(model.nsensor):
#     name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
#     stype = model.sensor_type[i]

#     if "cube" in name and stype in (
#         mujoco.mjtSensor.mjSENS_FRAMEPOS,
#         mujoco.mjtSensor.mjSENS_FRAMEQUAT,
#         mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
#         mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
#     ):
#         start = model.sensor_adr[i]
#         dim = model.sensor_dim[i]
#         sensor_values[name] = data.sensordata[start : start + dim].copy()

# # 输出每个cube相关传感器的值
# for name, values in sensor_values.items():
#     print(f"{name}: {values}")

# # 合并为一维向量
# sensor_vector = np.concatenate(list(sensor_values.values()))
# print("Combined cube sensor vector:", sensor_vector)

# # %%
# # === Contact position ===  
# # Observation space: only the contact position
# hand_tip_names = ['ff_tip', 'mf_tip', 'rf_tip', 'th_tip']
# object_name = 'cube'
# contact_obs = np.zeros((len(hand_tip_names), 3))
# contact_counts = np.zeros(len(hand_tip_names), dtype=int)

# for i in range(data.ncon):
#     contact = data.contact[i]
#     geom1_id, geom2_id = contact.geom1, contact.geom2
#     body1_id = model.geom_bodyid[geom1_id]
#     body2_id = model.geom_bodyid[geom2_id]
    
#     # 处理可能的 bytes 类型名称
#     body1_name = model.body(body1_id).name.decode() if isinstance(model.body(body1_id).name, bytes) else model.body(body1_id).name
#     body2_name = model.body(body2_id).name.decode() if isinstance(model.body(body2_id).name, bytes) else model.body(body2_id).name

#     # 判断是否为手指尖和 cube 的 contact
#     if ((body1_name in hand_tip_names and body2_name == object_name) or
#         (body2_name in hand_tip_names and body1_name == object_name)):
#         if body1_name in hand_tip_names:
#             idx = hand_tip_names.index(body1_name)
#         else:
#             idx = hand_tip_names.index(body2_name)
#         # 累加当前 contact 的位置
#         contact_obs[idx] += contact.pos.copy()
#         contact_counts[idx] += 1

# # 对有多个 contact 的手指，计算平均位置
# for idx in range(len(hand_tip_names)):
#     if contact_counts[idx] > 0:
#         contact_obs[idx] /= contact_counts[idx]

# print("Contact observation (averaged if multiple contacts):\n", contact_obs)
# print("\n")

# # %%
# # Contact detailed information
# # The more detailed information is used for force closure reward.
# hand_tip_names = ['ff_tip', 'mf_tip', 'rf_tip', 'th_tip']
# object_name = 'cube'

# for i in range(data.ncon):
#     contact = data.contact[i]
#     geom1_id, geom2_id = contact.geom1, contact.geom2
#     body1_id = model.geom_bodyid[geom1_id]
#     body2_id = model.geom_bodyid[geom2_id]
    
#     # 处理可能的 bytes 类型名称
#     body1_name = model.body(body1_id).name.decode() if isinstance(model.body(body1_id).name, bytes) else model.body(body1_id).name
#     body2_name = model.body(body2_id).name.decode() if isinstance(model.body(body2_id).name, bytes) else model.body(body2_id).name

#     # 筛选出与手指尖和 cube 的接触
#     if ((body1_name in hand_tip_names and body2_name == object_name) or
#         (body2_name in hand_tip_names and body1_name == object_name)):
        
#         # 获取 contact 位置
#         pos = contact.pos.copy()
        
#         # 获取 contact frame（存储为 9 个数，行主序，即第一行是 contact normal 等）
#         # 重塑为 3×3 的矩阵，其中行分别为 X（normal）、Y 和 Z 轴
#         frame = contact.frame.copy().reshape(3, 3)
        
#         # 利用 mj_contactForce 将 contact 的 force 转换为更直观的 3D 力和 3D 扭矩
#         force_torque = np.zeros(6, dtype=np.float64)
#         mujoco.mj_contactForce(model, data, i, force_torque)
#         force = force_torque[:3]
#         torque = force_torque[3:]
        
#         # 根据哪个手指参与 contact 进行打印或存储
#         if body1_name in hand_tip_names:
#             fingertip = body1_name
#         else:
#             fingertip = body2_name
        
#         print("Contact index:", i)
#         print("Fingertip:", fingertip)
#         print("Contact position:", pos)
#         print("Contact frame:\n", frame)
#         print("Contact force:", force)
#         print("Contact torque:", torque)
#         print("-" * 50)
# # %%
# """
# passive launch with steps
# """
# # viewer=mujoco.viewer.launch_passive(model, data)
# # cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")
# # viewer.cam.fixedcamid = cam_id
# # viewer.cam.type = 2
# # for _ in range(1000):
    
# #     mujoco.mj_step(model, data)
# #     viewer.sync()
# #     print(data.sensordata)
# #     time.sleep(0.001)

# """
# active launch
# """
# # viewer = mujoco.viewer.launch(model, data)
# # %%
