from tkinter import S
import numpy as np
import cv2
import time
import gymnasium as gym
from gymnasium import spaces
 
from gymnasium.core import ActionWrapper
import numpy as np
from gymnasium import spaces
import os

from numpy.core.defchararray import count
import rospy
from PIL import Image
import math
from collections import deque

from franka_utils import *
import pyrealsense2 as rs

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
import time
import logging

from franka_interface import ArmInterface, RobotEnable, GripperInterface
# ids camera lib for use of IDS ueye cameras.
# https://www.ids-imaging.us/files/downloads/ids-peak/readme/ids-peak-linux-readme-1.2_EN.html
#import ids
import time
import signal
import cv2
import multiprocessing
from gymnasium.spaces import Box as GymBox

from jsac.envs.franka_rw.gdino import create_mask, g_dino_inference
from groundingdino.util.inference import load_model
 
from jsac.envs.franka_rw.detic import Detic, detic_inference 

NPT = 50
DISTANCE_THRESHOLD = 0.13
TARGET_X_BOUNDARY = 0.2
TARGET_Y_BOUNDARY = 0.2
HEIGHT = 450
WIDTH = 800


T_ec = [[-6.69290020e-03,  9.97499486e-01, -6.95722171e-02,  5.92557060e-02],
        [-9.99811234e-01, -5.94550266e-03,  1.02341487e-02,  7.33507238e-05],
        [ 9.81631686e-03,  6.96436833e-02,  9.97364711e-01, -7.08931174e-02],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

OBJECT_POS = np.array([
    [0.471, -0.127, 0.05], ## ITEM 1 [pear]  
    [0.471, 0.176, 0.05],  ## ITEM 2 [mouse]
    [0.726, 0.158, 0.05],  ## ITEM 3 [apple] 
    [0.720, -0.123, 0.05], ## ITEM 4 [eyeglasses] 
], dtype=np.float32)

OBJECT_NAMES = ["green pear", "wireless mouse", "white baseball", "eyeglasses"]


class FrankaPanda_Visual_Reacher(gym.Env):
    """
    Gym env for the real franka robot. Set up to perform the placement of a peg that starts in the robots hand into a slot
    """
    def __init__(self, dt=0.04, classifier='gdino', image_width=WIDTH, image_height=HEIGHT, seed=9, size_tol=0.45):
        np.random.seed(seed)
        self.DT= dt
        self.dt = dt
        self.ep_time = 0
        signal.signal(signal.SIGINT, self.close)
        # config_file = os.path.join(os.path.dirname(__file__), os.pardir, 'reacher.yaml')
        self.conficlassifier_step = configure('/home/chemist/projects/RLC/JSAC/jsac/envs/franka_rw/reacher.yaml')
        self.conf_exp = self.conficlassifier_step['experiment']
        self.conf_env = self.conficlassifier_step['environment']
        rospy.init_node("franka_robot_gym")
        self.init_joints_bound = self.conf_env['reset-bound']
        #self.target_joints = self.conf_env['target-bound']
        self.safe_bound_box = np.array(self.conf_env['safe-bound-box'])
        self.target_box = np.array(self.conf_env['target-box'])
        self.joint_angle_bound = np.array(self.conf_env['joint-angle-bound'])
        self.return_point = self.conf_env['return-point']
        self.out_of_boundary_flag = False
        self.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']

        self.robot = ArmInterface(True)
        force = 1e-6
        self.robot.set_collision_threshold(cartesian_forces=[force,force,force,force,force,force])
        self.robot_status = RobotEnable()
        self.control_frequency = 1/dt
        self.rate = rospy.Rate(self.control_frequency)

        self.classifier = classifier
        
        if self.classifier == 'gdino':
            self.classifier_model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", 
                                               "../GroundingDINO/asset/groundingdino_swinb_cogcoor.pth")
        else:
            vocabs = OBJECT_NAMES
            self.classifier_model = Detic(vocabs)

        self._size_tol = size_tol

        self.ct = dt
        self.tv = time.time()

        self._image_width = image_width
        self._image_height = image_height

        self.joint_states_history = deque(np.zeros((5, 21)), maxlen=5)
        self.torque_history = deque(np.zeros((5, 7)), maxlen=5)
        self.last_action_history = deque(np.zeros((5, 7)), maxlen=5)
        self.time_out_reward = False
        action_dim = 7
        self.prev_action = np.zeros(action_dim)
        self.obs_image = None
        
        self.target = 0

        self.x_intervals = [(0.25, 0.93),   (0.93, 0.93),  (0.25, 0.93), (0.25, 0.25)]
        self.y_intervals = [(-0.53, -0.53), (-0.53, 0.53), (0.53, 0.53), (-0.53, 0.53)]
        self.z_intervals = [(0.002, 0.8),   (0.002, 0.8),  (0.002, 0.8), (0.002, 0.8)]

        self.create_all_points()

        # self.reward_functions = {'default': self.get_reward}

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline_profile = self.pipeline.start(config)

        for i in range(10):
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
        
        self.previous_place_down = None

        self.joint_action_limit = 0.275

        self.action_space = GymBox(low=-self.joint_action_limit * np.ones(7), high=self.joint_action_limit*np.ones(7))
        self.joint_angle_low = [j[0] for j in self.joint_angle_bound]
        self.joint_angle_high = [j[1] for j in self.joint_angle_bound]

        self.observation_space = GymBox(
            low=np.array(
                self.joint_angle_low  # q_actual
                + list(-np.ones(7)*self.joint_action_limit)  # qd_actual
                + list(-np.ones(7)*self.joint_action_limit)  # previous action in cont space
            ),
            high=np.array(
                self.joint_angle_high  # q_actual
                + list(np.ones(7)*self.joint_action_limit)  # qd_actual
                + list(np.ones(7)*self.joint_action_limit)    # previous action in cont space
            )
        )

        self.image_space = GymBox(low=0., high=255., 
        shape=[self._image_height, self._image_width, 3], dtype=np.uint8)

        self.total_timesteps = 0
        self.robot.exit_control_mode(0.2)

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def monitor(self, reward, done, info):
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if done:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self, 'get_normalized_score'):
                info['episode']['return'] = self.get_normalized_score(
                    info['episode']['return']) * 100.0
        return info
        
    def get_robot_jacobian(self):
        return self.robot.zero_jacobian()
 
    def euler_from_quaternion(self,q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        w,x,y,z = q        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

    def get_state(self):
        joint_angles = extract_values(self.robot.joint_angles(), self.joint_names)
        joint_velocitys = extract_values(self.robot.joint_velocities(), self.joint_names)
        ee_pose = self.robot.endpoint_pose()
        self.ee_position = ee_pose['position']
        ee_quaternion = [ee_pose['orientation'].w, ee_pose['orientation'].x,
                         ee_pose['orientation'].y, ee_pose['orientation'].z]
        
        self.distance = np.sqrt(np.sum((self.ee_position - self.target_pos)**2))

        if self.classifier_step > 0 and self.classifier_step % 20 == 0:
            print("{:.2f}".format(self.distance))
        else: 
            print("{:.2f}".format(self.distance), end=', ')
 
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        image = np.asanyarray(color_frame.get_data())
        image = cv2.resize(image, (WIDTH, HEIGHT))
        
        if not hasattr(self, 'camera_matrix'):
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            self.camera_matrix = np.array([
                    [intrinsics.fx, 0, intrinsics.ppx],
                    [0, intrinsics.fy, intrinsics.ppy],
                    [0, 0, 1]
                ], dtype=np.float32)
            self.dist_coeffs = np.array([])

        quat_new = [ee_pose['orientation'].x, ee_pose['orientation'].y,
                         ee_pose['orientation'].z, ee_pose['orientation'].w]
        t_e_new, R_e_new = self.ee_position, tf.transformations.quaternion_matrix(quat_new)[0:3, 0:3]
        
        T_be_new = np.eye(4)
        T_be_new[:3, :3] = R_e_new
        T_be_new[:3, 3] = t_e_new

        T_bc_new = T_be_new @ T_ec
        T_cb_new = np.linalg.inv(T_bc_new)
        R_new = T_cb_new[:3, :3]
        self.tvec = T_cb_new[:3, 3].reshape(3, 1)
        self.rvec, _ = cv2.Rodrigues(R_new)
        
        if self.classifier == 'gdino':
            xyxy, self.classifier_time = g_dino_inference(image, self.classifier_model, self.target_name)
        else:
            xyxy, self.classifier_time = detic_inference(image, self.classifier_model, self.target_name)
        self.current_mask = np.zeros((HEIGHT,  WIDTH), dtype=np.uint8) 
        self.gdino_center = (-2000, -2000)

        if xyxy is not None and xyxy.size != 0:
            if not self.check_in_region(xyxy):
                self.current_mask, self.gdino_center = create_mask(self.current_mask.copy(), xyxy=xyxy)

        target, _ = cv2.projectPoints(self.target_pos, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        
        target = target.squeeze(0).reshape(1, 2) 
        self.target_x, self.target_y = int(target[0, 0] * (WIDTH/1280.0)), int(target[0, 1] * (HEIGHT / 720.0))

        gt_center = self.target_x, self.target_y
        
        if self.is_classifier_prediction_accurate(gt_center, self.gdino_center, self.distance):
            self.gdino_num_accurate += 1
    
        # print(f'inference_time: {self.classifier_time}, target: {self.target_name}, accuracy: {accurate}, gt_center: {gt_center}, gdino_center: {self.gdino_center}')

        self.classifier_step += 1
        self.classifier_accuracy = float(self.gdino_num_accurate) / self.classifier_step
        
        white_pixels = float(np.sum(self.current_mask == 255))
        total_pixels = float(self.current_mask.size)
        self.mask_size = (white_pixels / total_pixels)  

        # im = image.copy() 
        # cv2.circle(im, (self.target_x, self.target_y), 3, (0, 0, 255), -1)
        # cv2.imshow('W1', im)
        # cv2.waitKey(1)
 
        self.current_image = np.dstack((image, self.current_mask))    

        # im = image.copy() 
        # cv2.circle(im, (self.target_x, self.target_y), 3, (0, 0, 255), -1)
        # extra = np.repeat(self.current_mask[..., np.newaxis], 3, axis=-1)
        # im = np.concatenate((im, extra), axis=1)
        # cv2.imshow("w1", im) 
        # cv2.waitKey(1)
        # cv2.imwrite(f'/home/chemist/projects/imclassifier_step/0/{self.im_index}.png', im)
        # self.im_index += 1

        # pos_str = np.array2string(np.array(self.ee_position), formatter={'float_kind': lambda x: f"{x:.3f}"})
        # quat_str = np.array2string(np.array(quat_new), formatter={'float_kind': lambda x: f"{x:.3f}"})

        # x, y, z, w = np.array(quat_new)
        # roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        # pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
        # yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        # euler_deg = np.array([math.degrees(roll), math.degrees(pitch), math.degrees(yaw)])
        # euler_str = np.array2string(euler_deg, formatter={'float_kind': lambda x: f"{x:.3f}"})

        # print('position', pos_str, 'ee_quaternion', quat_str, 'Euler angles (deg)', euler_str)


        self.last_action_history.append(self.prev_action)
        
        observation = {
            'image': self.current_image,
            'last_action': self.prev_action,
            'joints': np.array(joint_angles),
            'joint_vels': np.array(joint_velocitys)
        }

        self.ee_orientation = ee_quaternion
        return observation

    def out_of_boundaries(self):
        x, y, z = self.robot.endpoint_pose()['position']
        
        x_bound = self.safe_bound_box[0,:]
        y_bound= self.safe_bound_box[1,:]
        z_bound = self.safe_bound_box[2,:]
        if scalar_out_of_range(x, x_bound):
            # print('x out of bound, motion will be aborted! x {}'.format(x))
            return True
        if scalar_out_of_range(y, y_bound):
            # print('y out of bound, motion will be aborted! y {}'.format(y))
            return True
        if scalar_out_of_range(z, z_bound):
            # print('z out of bound, motion will be aborted!, z {}'.format(z))
            return True
        return False

    def apply_joint_vel(self, joint_vels):
        joint_vels = dict(zip(self.joint_names, joint_vels))
        self.robot.set_joint_velocities(joint_vels)
        
        return True

    def reset(self):
        """
        reset robot to random pose
        Returns
        -------
        object
            Observation of the current state of this env in the format described by this envs observation_space.
        """
        self.target = random.choice(list(range(4)))
        # self.target = 0
        self.target_pos = OBJECT_POS[self.target]
        self.target_name = OBJECT_NAMES[self.target]

        print(f'\n\n** Target name: {self.target_name}')
        
        self.time_steps = 0
        self.ep_time = 0
        self.gdino_num_accurate = 0
        self.classifier_step = 0
        self.robot_status.enable()
        self.reset_ee_quaternion = [0,-1.,0,0]
        # stop the robot
        self.apply_action_multiple(np.zeros((7,)), 2)

        self._reset_stats()

        self.out_of_boundary_flag = False

        reset_pose = dict(zip(self.joint_names, self.return_point))
        reset_pose['panda_joint1'] += np.random.uniform(-0.05, 0.05)
        reset_pose['panda_joint4'] += np.random.uniform(-0.15, 0.15)
        reset_pose['panda_joint6'] += np.random.uniform(-0.15, 0.15)


        smoothly_move_to_position_vel(self.robot, self.robot_status, reset_pose, MAX_JOINT_VELs=1.3)
        smoothly_move_to_position_vel(self.robot, self.robot_status, reset_pose, MAX_JOINT_VELs=1.3)

        # stop the robot
        self.apply_action_multiple(np.zeros((7,)), 2)

        # get the observation
        obs_robot = self.get_state()

        prop = np.concatenate((obs_robot["joints"], [0]*7))

        self.time_steps = 0
        self.tv = time.time()
        self.reset_time = time.time()

        self.cur_step = 0 
        self._reset_stats()
        return (obs_robot['image'].copy(), prop.copy())


    def step(self, action, pose_vel_limit=0.3):
        self.ep_time += self.dt
        self.cur_step += 1
        self.robot_status.enable()
        
        # limit joint action
        action = action.reshape(-1)
        action = action * self.joint_action_limit

        # convert joint velocities to pose velocities
        pose_action = np.matmul(self.get_robot_jacobian(), action)

        # limit action
        pose_action[:3] = np.clip(pose_action[:3], -pose_vel_limit, pose_vel_limit)

        # safety
        out_boundary = self.out_of_boundaries()
        pose_action[:3] = self.safe_actions(pose_action[:3])

        # calculate joint actions
        d_angle =  np.array(self.euler_from_quaternion(self.reset_ee_quaternion)) - np.array(self.euler_from_quaternion(self.ee_orientation))
        for i in range(3):
            if d_angle[i] < -np.pi:
                d_angle[i] += 2*np.pi
            elif d_angle[i] > np.pi:
                d_angle[i] -= 2*np.pi

        d_X = pose_action
        
        if out_boundary: 
            # print('Out of Boundary!!  EE Pos:', self.ee_position)
            d_X[3:] = 0
            action = self.get_joint_vel_from_pos_vel(d_X) 

        action = self.handle_joint_angle_in_bound(action)
        self.apply_action_multiple(action, 2) 
        self.prev_action = action

        done = False
        
        delay = (self.ep_time + self.reset_time) - time.time()
        if delay > 0:
            time.sleep(np.float64(delay))

        # get next observation
        observation_robot = self.get_state()

        self.time_steps += 1
        
        # construct the state
        prop = np.concatenate((observation_robot["joints"], action))
         
        image = observation_robot['image'].copy()
        prop = prop.copy()
        reward = self.calculate_img_reward(self.mask_size) - 1
        done = False
        info = {}

        if self.distance < DISTANCE_THRESHOLD and self.check_target_in_boundary():
            done = True
            reward = 5
            self.apply_action_multiple(np.zeros((7,)), 5)
        
        info = {
            'cf_step':self.classifier_step, 
            'cf_time':self.classifier_time, 
            'cf_accuracy':self.classifier_accuracy,
            'target_object':self.target
        }

        return (image, prop), reward, done, info

    def handle_joint_angle_in_bound(self, action):
        current_joint_angle = self.robot.joint_angles()
        in_bound = [False] * 7
        for i, joint_name in enumerate(self.joint_names):
            if current_joint_angle[joint_name] > 0.05 + self.joint_angle_bound[i][1]:
                 
                action[i] = -0.5
            elif current_joint_angle[joint_name] < -0.05+ self.joint_angle_bound[i][0]:
                action[i] = +0.5
        return action

    def get_timeout_reward(self):
        if self.time_out_reward:
            reward = -1
            print('call time out reward {:+.3f}'.format(reward))
            return reward
        else:
            return 0

    def move_to_pose_ee(self, ref_ee_pos, pose_vel_limit=0.2):
        counter = 0
        
        while True:
            self.robot_status.enable()
            counter += 1
            #action = agent.act(observations['ee_states'], ref_ee_pos, self.get_robot_jacobian(), add_noise=False)
            self.get_state()
            action = np.zeros((4,))
            action[:3] = ref_ee_pos-self.ee_position
            action[-1] = 1
            
            #if max(np.abs(action[:3])) < 0.005 or 
            #print(action)
            if max(np.abs(action[:3])) < 0.005 or counter > 100:
                break

            #self.step(action, ignore_safety=True)
            # limit action
            pose_action = np.clip(action[:3], -pose_vel_limit, pose_vel_limit)

            # calculate joint actions
            d_angle =  np.array(self.euler_from_quaternion(self.reset_ee_quaternion)) - np.array(self.euler_from_quaternion(self.ee_orientation))
            for i in range(3):
                if d_angle[i] < -np.pi:
                    d_angle[i] += 2*np.pi
                elif d_angle[i] > np.pi:
                    d_angle[i] -= 2*np.pi
            d_angle *= 0.5
            #print('d_angle', d_angle)
            d_X = np.array([pose_action[0], pose_action[1], pose_action[2], d_angle[0],d_angle[1],d_angle[2]])
            joints_action = self.get_joint_vel_from_pos_vel(d_X)
            # print('joints_action', joints_action)
            self.apply_joint_vel(joints_action)
            
            # action cycle time
            self.rate.sleep()
        self.apply_joint_vel(np.zeros((7,)))

    def get_joint_vel_from_pos_vel(self, pose_vel):
        return np.matmul(np.linalg.pinv( self.get_robot_jacobian() ), pose_vel)

    def safe_actions(self, action):
        out_boundary = self.out_of_boundaries()
        x, y, z = self.robot.endpoint_pose()['position']
        self.box_Normals = np.zeros((6,3))
        self.box_Normals[0,:] = [1,0,0]
        self.box_Normals[1,:] = [-1,0,0]
        self.box_Normals[2,:] = [0,1,0]
        self.box_Normals[3,:] = [0,-1,0]
        self.box_Normals[4,:] = [0,0,1]
        self.box_Normals[5,:] = [0,0,-1]
        self.planes_d = [   self.safe_bound_box[0][0],
                            -self.safe_bound_box[0][1],
                            self.safe_bound_box[1][0],
                            -self.safe_bound_box[1][1],
                            self.safe_bound_box[2][0],
                            -self.safe_bound_box[2][1]]
        if out_boundary:
            action = np.zeros((3,))
            for i in range(6):
                action += 0.1 * self.box_Normals[i] * ( (self.box_Normals[i].dot(np.array([x,y,z])) - self.planes_d[i]) < 0 ) 

        return action
    
    def is_classifier_prediction_accurate(self, gt_pos, classifier_pos, dist):
        mlt = (-0.5 * dist) + 1.5 
        acceptable_dist = (WIDTH / 6) * mlt 
        if isinstance(acceptable_dist, np.ndarray):
            acceptable_dist = acceptable_dist[0]
        distance = math.sqrt((gt_pos[0] - classifier_pos[0])**2 + (gt_pos[1] - classifier_pos[1])**2)

        ret = False
        if classifier_pos[0] == -2000 and classifier_pos[1] == -2000:
            w_tol = WIDTH * 0.025
            h_tol = HEIGHT * 0.025
            if (gt_pos[0] < w_tol or gt_pos[0] > WIDTH - w_tol) or (gt_pos[1] < h_tol or gt_pos[1] > HEIGHT - h_tol):
                return True
            
            return False

        ret = distance < acceptable_dist

        return ret
    
    def create_points(self, xmin, xmax, ymin, ymax, zmin, zmax):
        if np.isclose(zmin, zmax):
            x_coords = np.linspace(xmin, xmax, NPT, dtype=np.float64)
            y_coords = np.linspace(ymin, ymax, NPT, dtype=np.float64)
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            z_grid = np.full_like(x_grid, zmin, dtype=np.float64)
            return np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
        p1 = np.array([xmin, ymin, zmin], dtype=np.float64)
        p2 = np.array([xmax, ymax, zmin], dtype=np.float64)
        line_points = np.linspace(p1, p2, NPT, dtype=np.float64)
        z_values = np.linspace(zmin, zmax, NPT, dtype=np.float64)
        x = np.repeat(line_points[:, 0], NPT).astype(np.float64)
        y = np.repeat(line_points[:, 1], NPT).astype(np.float64)
        z = np.tile(z_values, NPT).astype(np.float64)
        return np.column_stack((x, y, z))

    def create_all_points(self):
        x_intervals = self.x_intervals
        y_intervals = self.y_intervals
        z_intervals = self.z_intervals
        pts = [np.array(self.create_points(xmin, xmax, ymin, ymax, zmin, zmax), dtype=np.float64) for (xmin, xmax), (ymin, ymax), (zmin, zmax) in zip(x_intervals, y_intervals, z_intervals)]
        self.excluded_points = np.vstack(pts).astype(np.float64)
        print('excluded_points shape: ', self.excluded_points.shape)
    
    def check_in_region(self, xyxy): 
        return 
        projected_points, _ = cv2.projectPoints(self.excluded_points, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        points = projected_points.reshape(-1, 2)
        
        x1, y1, x2, y2 = map(int, xyxy[:4])
        x_all, y_all = points[:, 0], points[:, 1]

        x_all, y_all  = (x_all * (WIDTH/1280.0)), (y_all * (HEIGHT / 720.0))
    
        x_all = np.clip(x_all, -2000, 2000)
        y_all = np.clip(y_all, -2000, 2000)

        # im = self.imt.copy()
        # for (u_proj, v_proj) in zip(x_all, y_all):
        #     u_p, v_p = int(u_proj), int(v_proj)
        #     print(u_p, v_p)
        #     cv2.circle(im, (u_p, v_p), 1, (0, 0, 255), -1)
        
        # cv2.imshow('W1', im)

        mask = (x_all >= x1) & (x_all < x2) & (y_all >= y1) & (y_all < y2)
        if mask.any():
            return True
        return False
    
    def apply_action_multiple(self, action, times):
        for _ in range(times):
            self.apply_joint_vel(action)

    def calculate_img_reward(self, perc):        
        return (2.0/(1+np.exp(-perc*10.0))) - 1.0 
    
    def check_target_in_boundary(self):
        w, h = self._image_width, self._image_height
        x, y = int(self.target_x), int(self.target_y)
        
        min_x = w * TARGET_X_BOUNDARY
        max_x = w * (1 - TARGET_X_BOUNDARY)
        min_y = h * TARGET_Y_BOUNDARY
        max_y = h * (1 - TARGET_Y_BOUNDARY)

        if min_x <= x <= max_x and min_y <= y <= max_y:
            return True
    
        return False

    def close(self, signum=None, frame=None):
        self.apply_action_multiple(np.zeros((7,)), 5)
        self.pipeline.stop()
        # exit(1)

    def seed(self, seed):
        np.random.seed(seed)
        
        

# if __name__ == "__main__":
#     env = FrankaPanda_Visual_Reacher(dt=0.15)

#     env.reset()

#     mx = 0
#     t1 = time.time()
#     prev_ee = env.ee_position

#     for i in range(80):
#         # action = np.random.uniform(size=env.action_space.shape[0])
#         action = np.zeros((7,))
#         action[3] = -0.06
#         action[5] = -0.06
#         env.step(action)
#         t2 = time.time()
#         print('step_time: ', t2-t1)    
#         t1 = t2
#         new_ee = env.ee_position

#         a, b = prev_ee, new_ee
#         prev_ee = new_ee
#         change = ((b[0] - a[0])**2 + (b[1] - a[1])**2 + (b[2] - a[2])**2)**0.5
#         mx = max(mx, change)

#     env.apply_joint_vel(np.zeros((7,)))
#     time.sleep(0.05)
#     env.apply_joint_vel(np.zeros((7,)))
#     env.close()