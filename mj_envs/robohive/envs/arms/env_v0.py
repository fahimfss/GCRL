import time
import copy
import math

import random
import cv2 as cv
import collections
import numpy as np
from PIL import Image
import gymnasium as gym
import multiprocessing as mp
from robohive.envs import env_base_0

from robohive.utils.quat_math import mat2euler
from robohive.envs.arms.python_api_2 import BodyIdInfo, get_touching_objects, ObjLabels
from robohive.envs.arms.gdino import create_mask, g_dino_inference, async_g_dino_inference
    
from groundingdino.util.inference import load_model
from robohive.envs.arms.detic import Detic, detic_inference
from robohive.envs.arms.owlv2 import OwlV2, owlv2_inference

# MASK_SIZE_LIMIT = 30
# MASK_SIZE_LIMIT_DIST = 30
DISTANCE_THRESHOLD = 0.1
TARGET_X_BOUNDARY = 0.2
TARGET_Y_BOUNDARY = 0.2
N = 40


class EnvV0(env_base_0.MujocoEnv):
    DEFAULT_OBS_KEYS = ['qp_robot', 'prev_action']
    DEFAULT_PROPRIO_KEYS = ['qp_robot', 'prev_action']

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)
        if 'franka' in model_path:
            self.robot_name = 'franka'
        else:
            self.robot_name = 'ur10'
        self._setup(**kwargs)


    def _setup(self,
               robot_site_name,
               image_width=960,
               image_height=540,
               frame_skip = 20, 
               env_mode = "train",          # "train", "eval_ofd", "eval", "inference_1", "inference_3"
               reward_mode = "mask_size",   # "distance", "mask_size"
               classifier = "gdino",        # "gdino"
               inference_type = "sync",     # "sync", "async"
               obs_keys=DEFAULT_OBS_KEYS,
               proprio_keys=DEFAULT_PROPRIO_KEYS,
               ofd_index=0,
               **kwargs,
        ):

        # ids 
        self.grasp_sid = self.sim.model.site_name2id(robot_site_name) #robot part name
        self.center_obj_range = np.array([[-0.15, 0.15], [0.29, 0.41]])
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height  
        self.fixed_positions = None
        self.cam_init = True
        self._setup_camera() 
        self.current_image = np.ones((image_height, image_width, 4), dtype=np.uint8) 
        self.current_mask = None
        self.target_x, self.target_y = 0, 0
        self.gdino_time = 0
        self.gdino_step = 0
        self.gdino_error = 0
        self.gdino_num_accurate = 0
        self.gdino_accuracy = 0
        self.gs = 0
        self.distance = 1.0
        self._target_in_boundary = False
        self.mask_size = 0
        self.TM = time.time()
        self.prev_action = np.array([0] * self.sim.model.nu)
        
        self.x_intervals = [(-0.7, 0.7),  (0.7, 0.7),   (-0.7, 0.7),  (-0.7, -0.7), (-0.7, 0.7)]
        self.y_intervals = [(0.87, 0.87), (-0.8, 0.87), (-0.8, -0.8), (-0.8, 0.87), (-0.8, 0.01)]
        self.z_intervals = [(0.83, 2.5),  (0.83, 2.5),  (0.83, 2.5),  (0.83, 2.5),  (0.83, 0.83)]
        self.create_all_points()
        
        self.env_mode = env_mode
        self.reward_mode = reward_mode
        
        print('Reward mode: ', reward_mode)
        
        self.classifier = classifier
        self.inference_type = inference_type 
        
        self.objects = {
            'object_1': 'red apple',
            'object_2': 'green block',
            'object_3': 'chocolate donut',
            'object_4': 'round bottomed flask',
            'object_5': 'yellow toy duck',
            'object_6': 'banana',
            'object_7': 'purple alarm clock',
            'object_8': 'pink cup',
            'object_9': 'water bottle',
            'object_10': 'light bulb',
            'object_11': 'wine glass',
            'object_12': 'copper bowl',
            'object_13': 'silver headphone',
            'object_14': 'hammer',
            'object_15': 'digital camera',
            'object_16': 'blue stapler',
            'object_17': 'white egg',
            'object_18': 'toy train',
            'object_19': 'teapot',
            'object_20': 'eyeglasses',
        }
        # ['red apple', 'green block', 'chocolate donut', 'round bottomed flask', 'yellow toy duck', 'banana', 'purple alarm clock', 'pink cup', 'water bottle', 'light bulb', 'wine glass', 'copper bowl', 'silver headphone', 'hammer', 'digital camera', 'blue stapler', 'white egg', 'toy train', 'teapot', 'eyeglasses']

        
        if reward_mode == "distance":
            weighted_reward_keys = {
                "distance": -1.0, 
                "contact": 0.,
                'penalty': 0,
                'mask_size': 0.,
                "done": 5.,
            }
        else:
            weighted_reward_keys = {
                "distance": 0., 
                "contact": 0.,
                'penalty': 1.,
                'mask_size': 1,
                "done": 5.,
            }
            
        if self.classifier == "gdino" and self.inference_type=="sync":
            self.classifier_model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", 
                                               "../GroundingDINO/asset/groundingdino_swinb_cogcoor.pth")
        elif self.classifier == "gdino" and self.inference_type=="async":
            self.image_queue = mp.Queue()
            self.mask_queue = mp.Queue()
            
            img_shape = (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)
            original_array = np.zeros(img_shape, dtype=np.uint8)
            self.img_shm = mp.shared_memory.SharedMemory(create=True, size=original_array.nbytes)
            self.img_arr = np.ndarray(img_shape, dtype=np.uint8, buffer=self.img_shm .buf)
            
            self.mask_process = mp.Process(target=async_g_dino_inference, 
                                           args=(img_shape, self.img_shm.name, 
                                            self.image_queue, self.mask_queue))
            self.mask_process.start()
            self.images_sent = 0
            self.masks_recieved = 0
        elif self.classifier == "detic":
            assert self.inference_type == "sync"
            # It is not implemented for async for now
            vocabs = [v for v in self.objects.values()]
            self.classifier_model = Detic(vocabs)
            
        elif self.classifier == "owlv2":
            assert self.inference_type == "sync"
            # It is not implemented for async for now
            vocabs = [v for v in self.objects.values()]
            self.classifier_model = OwlV2(vocabs)
        
        self.target_name = "red apple"
        self._min_dist = 10000
        self.mask_size_counter = 0
                
        self.ofd = {
            0: [2, 1, 0, 11, 12],
            1: [15, 18, 10, 13, 9],
            2: [3, 16, 6, 5, 19],
            3: [4, 14, 17, 7, 8],
            4: [9, 4, 18, 13, 6],
            5: [12, 2, 11, 5, 0],
            6: [16, 17, 19, 7, 8],
            7: [10, 15, 14, 1, 3],
            8: [9, 5, 19, 11, 18],
            9: [0, 16, 10, 14, 2],
            10: [8, 13, 12, 17, 4],
            11: [3, 1, 15, 6, 7],
            12: [1, 5, 15, 16, 13],
            13: [8, 7, 9, 6, 4],
            14: [12, 19, 0, 18, 10],
            15: [3, 17, 2, 14, 11],
            16: [11, 6, 7, 16, 13],
            17: [3, 4, 1, 14, 5],
            18: [19, 18, 0, 9, 8],
            19: [12, 17, 10, 2, 15]
        } 
        
        self.ofd_index = ofd_index
        print('ofd_index:', ofd_index)

        self.TS = list(self.objects.keys())
        self.TN = list(self.objects.values())

        self.target_sid = self.sim.model.site_name2id(self.TS[0]) 
        self.r = 2
        self.target_site_name = self.TS[0]
        self.epi_target_object_num = 0
        
        if 'target_obj_num' in kwargs:
            self.target_obj_num = kwargs['target_obj_num'] 
            kwargs.pop('target_obj_num')
            
        if 'step_time' in kwargs:
            self.step_time = kwargs['step_time'] 
            kwargs.pop('step_time')
        else:
            self.step_time = None

        super()._setup(obs_keys=obs_keys,
                       proprio_keys=proprio_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reward_mode="dense",
                       frame_skip=frame_skip,
                       **kwargs)
        self.init_qpos[:] = self.sim.model.key_qpos[0].copy()


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['qp_robot'] = sim.data.qpos[:sim.model.nu].copy()
        obs_dict['qv_robot'] = sim.data.qvel[:sim.model.nu].copy()
        obs_dict['prev_action'] = self.prev_action
        obs_dict['xmat_pinch'] = mat2euler(np.reshape(self.sim.data.site_xmat[self.grasp_sid], (3, 3)))
        obs_dict['claw_ori_err'] = obs_dict['xmat_pinch'] - np.array([-np.pi, 0, -np.pi/2])
        obs_dict['reach_err'] = sim.data.site_xpos[self.target_sid]-sim.data.site_xpos[self.grasp_sid]
        obs_dict['power_cost'] = sim.data.qvel.copy()*sim.data.qfrc_actuator.copy()
        obs_dict['mask_size'] = np.array([self.mask_size])  
        self.current_observation = self.get_observation()

        this_model = sim.model
        id_info = BodyIdInfo(this_model)
        this_data = sim.data

        touching_objects = set(get_touching_objects(this_model, this_data, id_info, self.target_site_name))

        obs_vec = self._obj_label_to_obs(touching_objects)
        obs_dict["touching_body"] = obs_vec

        return obs_dict
    
    def _obj_label_to_obs(self, touching_body):
        # Function to convert touching body set to an binary observation vector
        # order follows the definition in python_api file
        obs_vec = np.array([0, 0, 0])
        for i in touching_body:
            if i == ObjLabels.LEFT_GRIP:
                obs_vec[0] += 1
            elif i == ObjLabels.RIGHT_GRIP:
                obs_vec[1] += 1
            else:
                obs_vec[2] += 1

        return obs_vec
    
    def get_reward_dict(self, obs_dict):
        self.distance = np.linalg.norm(obs_dict['reach_err'], axis=-1)[0]
        
        if self.distance < self._min_dist:
            self._min_dist = self.distance
        
        mask_size_reward = np.array([self.calculate_img_reward(self.mask_size)])
        contact = np.array([np.sum(obs_dict["touching_body"][0][0][:2])])
        
        mask_size = int(self.mask_size * 100)
        # if self.reward_mode == 'mask_size': 
        #     if mask_size >= MASK_SIZE_LIMIT:
        #         self.mask_size_counter += 1
        #     done_1 = np.array([self.mask_size_counter]) == 5
        #     done_2 = self.mask_size_counter == 5
        # else:
        if self.distance < DISTANCE_THRESHOLD and self._target_in_boundary:
            done_1 = np.full((1,), True, dtype=np.bool_)
            done_2 = True
        else:
            done_1 = np.full((1,), False, dtype=np.bool_)
            done_2 = False
                
             
        rwd_dict = collections.OrderedDict((
            ('distance',  self.distance),
            ('contact', contact),
            ('penalty', np.array([-1])),  
            ('mask_size',  mask_size_reward),
            ('done', done_1),  
        )) 
         
        if self.env_mode == "train":
            rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        else:
            rwd_dict['dense'] = 1.0 if done_2 else 0
            rwd_dict['done'] = done_2
        
        return rwd_dict
    
    def place_object(self, obj_name, reset_qpos, x_pos=None, y_pos=None, drop=False):
        objec_bid = self.sim.model.body_name2id(obj_name)  # get body ID using object name
        object_jnt_adr = self.sim.model.body_jntadr[objec_bid]
        object_qpos_adr = self.sim.model.jnt_qposadr[object_jnt_adr]
        if drop: 
            reset_qpos[object_qpos_adr + 2] = -400
        else:
            reset_qpos[object_qpos_adr] = x_pos
            reset_qpos[object_qpos_adr + 1] = y_pos

        if obj_name == 'object_4':    
            objec_bid = self.sim.model.body_name2id('base_rbf')
            object_jnt_adr = self.sim.model.body_jntadr[objec_bid]
            object_qpos_adr = self.sim.model.jnt_qposadr[object_jnt_adr]
            if drop: 
                reset_qpos[object_qpos_adr + 2] = -400
            else:
                reset_qpos[object_qpos_adr] = x_pos
                reset_qpos[object_qpos_adr + 1] = y_pos
    
    def reset(self, reset_qpos=None, **kwargs): 
        # print("-->", self.gdino_num_accurate, self.gs, self.gdino_accuracy)

        print('min dist: ', self._min_dist)
        self._min_dist = 10000
        
        self.prev_action = np.array([0] * self.sim.model.nu)
        self.current_mask = None
        self.gdino_error = 0
        self.gdino_num_accurate = 0
        self.gdino_accuracy = 0
        self.distance = 1.0
        self._target_in_boundary = False
        self.gs = 0
        self.mask_size = 0
        self.mask_size_counter = 0
        
        if self.classifier == "gdino" and self.inference_type == "async":
            while(self.images_sent > self.masks_recieved):
                self.mask_queue.get()
                self.masks_recieved += 1
            self.images_sent = 0
            self.masks_recieved = 0
        
        ofd_items = self.ofd[self.ofd_index]
        train_items = list(set(range(20)) - set(ofd_items))
        
        if self.env_mode == "train" or self.env_mode == "eval":
            number = random.choice(train_items)
        elif self.env_mode == "eval_ofd":
            if 'object_id' in kwargs:
                number = kwargs['object_id']
                kwargs.pop('object_id')
            else:
                number = number = random.choice(list(range(20)))
            print(f'Object Id: {number}')
        else:
            number = self.target_obj_num 
        
        self.epi_target_object_num = number
        reset_qpos = self.sim.model.key_qpos[0].copy()
        
        self.target_name = self.TN[number] 
        self.target_site_name = self.TS[number] 
        
        self.target_sid = self.sim.model.site_name2id(self.target_site_name) 

        if self.env_mode == "inference_1" or self.env_mode == "inference_3":
            if self.env_mode == "inference_3":
                other_indices = random.sample([i for i in range(20) if i != number], 2)
                other_site_names = [self.TS[ind] for ind in other_indices]
                print('other sites:', other_site_names)
            else:
                other_site_names = []
            
            for obj_name in self.TS:
                if obj_name == self.target_site_name:
                    x_pos = 0
                    y_pos = 0.38

                    self.place_object(obj_name, reset_qpos, x_pos, y_pos)
                    
                    self.target_x = x_pos
                    self.target_y = y_pos
                elif obj_name in other_site_names:
                    if obj_name == other_site_names[0]:
                        x_pos = 0.2 + random.uniform(-0.03, 0.03)
                        y_pos = 0.35 + random.uniform(-0.02, 0.02)
                        self.place_object(obj_name, reset_qpos, x_pos, y_pos)
                    else:
                        x_pos = -0.2 + random.uniform(-0.03, 0.03)
                        y_pos = 0.35 + random.uniform(-0.02, 0.02)
                        self.place_object(obj_name, reset_qpos, x_pos, y_pos)
                else:
                    self.place_object(obj_name, reset_qpos, drop=True)
        
        elif self.env_mode == "eval_ofd" or self.env_mode == "eval" or self.env_mode == "train": 
            items = random.randint(2, 4)
            
            if self.env_mode == "eval_ofd":
                eval_items = list(set(range(20)))
                eval_items.remove(number)
                item_indices = random.sample(eval_items, items)
            else:
                train_items.remove(number)
                item_indices = random.sample(train_items, items)
                
            site_names = [self.TS[idx] for idx in item_indices]
            
            item_names = [self.TN[idx] for idx in item_indices]
            print('Other objects: ', item_names)
            
            site_names.append(self.target_site_name)
            random.shuffle(site_names)

            for obj_name in self.TS:
                if obj_name not in site_names:
                    self.place_object(obj_name, reset_qpos, drop=True)
            
            center_obj_x_pos = random.uniform(self.center_obj_range[0][0], 
                                              self.center_obj_range[0][1])
            
            center_obj_y_pos = random.uniform(self.center_obj_range[1][0], 
                                              self.center_obj_range[1][1])

            for index, obj_name in enumerate(site_names):
                x_pos = center_obj_x_pos + (((index - 2) * 0.165) +  random.uniform(-0.05, 0.05))
                y_pos = center_obj_y_pos + random.uniform(-0.075, 0.075)
                self.place_object(obj_name, reset_qpos, x_pos, y_pos)
         
        obs = super().reset(reset_qpos = reset_qpos, reset_qvel = None, **kwargs)
        
        site_pos = self.sim.data.site_xpos[self.target_sid]
        camera_matrix = self.compute_camera_matrix()
        self.target_x, self.target_y  = self.world_2_pixel(site_pos, camera_matrix) 
        site_pos[0] += 0.04
        rx, ry  = self.world_2_pixel(site_pos, camera_matrix) 
        try:
            self.r = math.sqrt((rx - self.target_x) ** 2 + (ry - self.target_y) ** 2)
        except:
            self.r = 0
        
        self.final_image = self.current_image
        
        self.TM = time.time()
        print("target:", self.target_name)
        return {'image': self.final_image, 'vector': obs}
    

    def get_observation(self):
        rgb = self.get_image_data()
        # print("rgb_shape: ", rgb.shape)
        site_pos = self.sim.data.site_xpos[self.target_sid].copy()
        camera_matrix = self.compute_camera_matrix()
        self.target_x, self.target_y = self.world_2_pixel(site_pos, camera_matrix) 
        self._target_in_boundary = self.check_target_in_boundary()
        
        site_pos[0] += 0.04
        rx, ry  = self.world_2_pixel(site_pos, camera_matrix) 
        try:
            self.r = math.sqrt((rx - self.target_x) ** 2 + (ry - self.target_y) ** 2)
        except:
            self.r = 0
        observation = {}
        observation["rgb"] = rgb
        
        return observation
    
    #setting a boundary of virtual box such that the arm will not accidentally
    def check_collision(self):
        """ Check if any joint is out of the defined boundary """
        if "ur10e" in self.sim.model.name: ## BOUNDARIES FOR UR10eEnv-v0
            x_min, x_max = -1.5, 1.5
            y_min, y_max = -1.7, 1.5
            z_min, z_max = 0.85, 2.23
            for i in range(1, 13):
                joint_frame_id = self.sim.model.jnt_bodyid[i]
                joint_pos = self.sim.data.xpos[joint_frame_id]
                if not (x_min <= joint_pos[0] <= x_max and 
                        y_min <= joint_pos[1] <= y_max and 
                        z_min <= joint_pos[2] <= z_max):
                    return True
        elif "franka" in self.sim.model.name:
            x_min, x_max = -3, 3
            y_min, y_max = -3, 3
            z_min, z_max = 0.85, 2.23
            for i in range(1, 9):
                joint_frame_id = self.sim.model.jnt_bodyid[i]
                joint_pos = self.sim.data.xpos[joint_frame_id]
                if not (x_min <= joint_pos[0] <= x_max and 
                        y_min <= joint_pos[1] <= y_max and 
                        z_min <= joint_pos[2] <= z_max):
                    return True

        return False
    
    def save_state(self):
        """ Save the current simulation state """
        self.previous_state = {
            'qpos': np.copy(self.sim.data.qpos),
            'qvel': np.copy(self.sim.data.qvel),
            'actuator': np.copy(self.sim.data.ctrl) if hasattr(self.sim.data, 'ctrl') else None
        }
    
    def restore_state(self, **kwargs):
        """ Restore the simulation state from self.previous_state """
        if self.previous_state:
            self.sim.data.qpos[:] = self.previous_state['qpos']
            self.sim.data.qvel[:] = self.previous_state['qvel']
            if self.previous_state['actuator'] is not None:
                self.sim.data.ctrl[:] = self.previous_state['actuator']
            obs = super().reset(reset_qpos = self.previous_state['qpos'], reset_qvel = None, **kwargs)
        return obs

    def step(self, a, **kwargs):
        """
        Step the simulation forward (t => t+1)
        Uses robot interface to safely step the forward respecting pos/ vel limits
        Accepts a(t) returns obs(t+1), rwd(t+1), done(t+1), info(t+1)
        change control method here if needed 
        """
        self.save_state()
        self.prev_action = a
        
        a = np.clip(a, self.action_space.low, self.action_space.high)

        last_pos = self.sim.data.qpos[:self.sim.model.nu].copy()
        if self.robot_name == 'franka' and hasattr(self, 'last_ctrl'):
            last_pos[-1] = self.last_ctrl[-1]
            
        self.last_ctrl = self.robot.step(ctrl_desired=a,
                                    last_qpos = last_pos,
                                    dt = self.dt,
                                    render_cbk=self.mj_render if self.mujoco_render_frames else None)


        if self.check_collision():
            # print("Collision detected, reverting action")
            self.restore_state()
            
        if self.step_time: 
            dt = time.time() - self.TM 
            if dt < self.step_time: 
                time.sleep(self.step_time - dt)
            self.TM = time.time()
     
        self.final_image = self.current_image

        return self.forward(self.final_image, **kwargs)
     
    
    def get_image_data(self, camera="end_effector_cam"):
        """
        Returns the RGB and depth images of the provided camera.

        Args:
            show: If True displays the images for five seconds or until a key is pressed.
            camera: String specifying the name of the camera to use.
        """ 
        # Initialize the simulator
        bgr = copy.deepcopy(
            self.sim.renderer.render_offscreen(width=self.IMAGE_WIDTH, 
                                               height=self.IMAGE_HEIGHT, 
                                               camera_id=camera, depth = False)) 

        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB) 
        
        if self.classifier == "gdino" and self.inference_type == "sync":
            xyxy, self.gdino_time = g_dino_inference(rgb, self.classifier_model, self.target_name, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
            # print(f'inference_time: {inference_time}')
            self.current_mask = np.zeros((self.IMAGE_HEIGHT,  self.IMAGE_WIDTH), dtype=np.uint8) 
            self.gdino_center = (-2000, -2000)
            
            if xyxy is not None and xyxy.size != 0:
                if not self.check_in_region(self.compute_camera_matrix(), xyxy):
                    self.current_mask, self.gdino_center = create_mask(self.current_mask.copy(), xyxy=xyxy)
        
            gt_center = int(self.target_x), int(self.target_y)
                    
            if self.is_classifier_prediction_accurate(gt_center, self.gdino_center, self.distance, self.IMAGE_WIDTH, self.IMAGE_HEIGHT):
                self.gdino_num_accurate += 1
                        
            self.gs += 1
            self.gdino_accuracy = float(self.gdino_num_accurate) / self.gs
            
            # print(self.gdino_num_accurate, self.gs, self.gdino_accuracy, self.gdino_time)

        elif self.classifier == "gdino" and self.inference_type == "async":
            if self.current_mask is None:
                np.copyto(self.img_arr, rgb)
                self.image_queue.put(self.target_name)
                self.images_sent += 1
                
                boxes, self.gdino_time, self.gdino_step = self.mask_queue.get() 
                mask = np.zeros((self.IMAGE_HEIGHT,  self.IMAGE_WIDTH), dtype=np.uint8)  
                self.current_mask, self.gdino_center = create_mask(mask, boxes=boxes)
                
                self.masks_recieved += 1
            else:
                if self.images_sent == 1:
                    np.copyto(self.img_arr, rgb)
                    self.image_queue.put(self.target_name)
                    self.images_sent += 1
                if not self.mask_queue.empty():
                    boxes, self.gdino_time, self.gdino_step = self.mask_queue.get() 
                    mask = np.zeros((self.IMAGE_HEIGHT,  self.IMAGE_WIDTH), dtype=np.uint8)  
                    self.current_mask, self.gdino_center = create_mask(mask, boxes=boxes)
                    
                    self.masks_recieved += 1
                    
                    np.copyto(self.img_arr, rgb)
                    self.image_queue.put(self.target_name)
                    self.images_sent += 1
                    
            gt_center = int(self.target_x), int(self.target_y)
            if self.is_classifier_prediction_accurate(gt_center, self.gdino_center, self.distance, self.IMAGE_WIDTH, self.IMAGE_HEIGHT):
                self.gdino_num_accurate += 1
            self.gs += 1
            self.gdino_accuracy = float(self.gdino_num_accurate) / self.gs
            
        elif self.classifier == "detic":
            xyxy, self.gdino_time = detic_inference(bgr, self.classifier_model, self.target_name)
            # print(f'inference_time: {inference_time}')
            self.current_mask = np.zeros((self.IMAGE_HEIGHT,  self.IMAGE_WIDTH), dtype=np.uint8) 
            self.gdino_center = (-2000, -2000)
            
            if xyxy is not None and xyxy.size != 0:
                if not self.check_in_region(self.compute_camera_matrix(), xyxy):
                    self.current_mask, self.gdino_center = create_mask(self.current_mask.copy(), xyxy=xyxy)
        
            gt_center = int(self.target_x), int(self.target_y)
                    
            if self.is_classifier_prediction_accurate(gt_center, self.gdino_center, self.distance, self.IMAGE_WIDTH, self.IMAGE_HEIGHT):
                self.gdino_num_accurate += 1
                        
            self.gs += 1
            self.gdino_accuracy = float(self.gdino_num_accurate) / self.gs

        elif self.classifier == "owlv2":
            xyxy, self.gdino_time = owlv2_inference(rgb, self.classifier_model, self.target_name)
            # print(f'inference_time: {inference_time}')
            self.current_mask = np.zeros((self.IMAGE_HEIGHT,  self.IMAGE_WIDTH), dtype=np.uint8) 
            self.gdino_center = (-2000, -2000)
            
            if xyxy is not None and xyxy.size != 0:
                if not self.check_in_region(self.compute_camera_matrix(), xyxy):
                    self.current_mask, self.gdino_center = create_mask(self.current_mask.copy(), xyxy=xyxy)
        
            gt_center = int(self.target_x), int(self.target_y)
                    
            if self.is_classifier_prediction_accurate(gt_center, self.gdino_center, self.distance, self.IMAGE_WIDTH, self.IMAGE_HEIGHT):
                self.gdino_num_accurate += 1
                        
            self.gs += 1
            self.gdino_accuracy = float(self.gdino_num_accurate) / self.gs
            


        x1, x2 = 0, self.IMAGE_WIDTH
        y1, y2 = 0, self.IMAGE_HEIGHT
        
        roi = self.current_mask[y1:y2, x1:x2]
        white_pixels = float(np.sum(roi == 255))
        total_pixels = float(roi.size)
        self.mask_size = (white_pixels / total_pixels)  

        self.current_image = np.dstack((rgb, self.current_mask))
        
        return np.array(np.fliplr(np.flipud(rgb)))

    def check_target_in_boundary(self):
        w, h = self.IMAGE_WIDTH, self.IMAGE_HEIGHT
        x, y = int(self.target_x), int(self.target_y)
        
        min_x = w * TARGET_X_BOUNDARY
        max_x = w * (1 - TARGET_X_BOUNDARY)
        min_y = h * TARGET_Y_BOUNDARY
        max_y = h * (1 - TARGET_Y_BOUNDARY)

        if min_x <= x <= max_x and min_y <= y <= max_y:
            return True
        
        return False
    def render(self, mode='rgb_array'):
        # Your implementation here, which should return an RGB array if mode is 'rgb_array'
        mode='rgb_array'
        if mode == 'rgb_array':
            rgb = copy.deepcopy(
            self.sim.renderer.render_offscreen(width=self.IMAGE_WIDTH, 
                                               height=self.IMAGE_HEIGHT, camera_id='end_effector_cam', 
                                               depth = False)
            )
            return rgb
        else:
            super().render(mode)

    def pixel_2_world(self, pixel_x, pixel_y, depth, width, height, camera="end_effector_cam"):
        """
        Converts pixel coordinates into world coordinates.

        Args:
            pixel_x: X-coordinate in pixel space.
            pixel_y: Y-coordinate in pixel space.
            depth: Depth value corresponding to the pixel.
            width: Width of the image (pixel).
            height: Height of the image (pixel).
            camera: Name of camera used to obtain the image.
        """

        if not self.cam_init:
            self.create_camera_data(width, height, camera)

        # Create coordinate vector
        pixel_coord = np.array([pixel_x, pixel_y, 1])

        # Apply the intrinsic matrix to get camera space coordinates
        pos_c = np.linalg.inv(self.cam_matrix) @ pixel_coord
        pos_c *= -depth  # Apply depth to scale to the actual position in camera space

        # Convert camera space coordinates to world coordinates
        pos_w = np.linalg.inv(self.cam_rot_mat) @ pos_c + self.cam_pos

        return pos_w

    def _setup_camera(self):
        """Sets up the camera to render the scene from the required view."""
        # This assumes you have a fixed camera in your model XML
        self.camera_id = self.sim.model.camera_name2id('end_effector_cam')
    
    def compute_camera_matrix(self, camera="end_effector_cam"):
        camera_id = self.sim.model.camera_name2id(camera)
        pos = np.array(self.sim.data.cam_xpos[camera_id], dtype=np.float64)
        rot_mat = np.array(self.sim.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
        fov = float(self.sim.model.cam_fovy[camera_id])

        translation = np.eye(4, dtype=np.float64)
        translation[0:3, 3] = -pos

        rotation = np.eye(4, dtype=np.float64)
        rotation[0:3, 0:3] = rot_mat.T

        focal_scaling = (1.0 / np.tan(np.deg2rad(fov) / 2.0)) * self.IMAGE_HEIGHT / 2.0
        focal = np.diag(np.array([-focal_scaling, focal_scaling, 1.0, 0], dtype=np.float64))[0:3, :]

        image = np.eye(3, dtype=np.float64)
        image[0, 2] = (self.IMAGE_WIDTH - 1) / 2.0
        image[1, 2] = (self.IMAGE_HEIGHT - 1) / 2.0
        return image @ focal @ rotation @ translation

    
    def franka_body_visible(self):
        for i in range(4):
            franka_body_sid = self.sim.model.site_name2id(f'bsite{i+1}') 
            franka_site_pos = self.sim.data.site_xpos[franka_body_sid]
            camera_matrix = self.compute_camera_matrix()
            x, y  = self.world_2_pixel(franka_site_pos, camera_matrix) 
            if x >= 0 and x < self.IMAGE_WIDTH and y >= 0 and y < self.IMAGE_HEIGHT:
                return True
        return False
    
    def is_classifier_prediction_accurate(self, gt_pos, classifier_pos, dist, image_width, image_height):
        mlt = (-0.5 * dist) + 1.5 
        acceptable_dist = (image_width / 7.5) * mlt 
        if isinstance(acceptable_dist, np.ndarray):
            acceptable_dist = acceptable_dist[0]
        distance = math.sqrt((gt_pos[0] - classifier_pos[0])**2 + (gt_pos[1] - classifier_pos[1])**2)
        
        if classifier_pos[0] == -2000 and classifier_pos[1] == -2000:
            w_tol = image_width * 0.025
            h_tol = image_height * 0.025
            if gt_pos[0] < w_tol and gt_pos[0] > image_width - w_tol and gt_pos[1] < h_tol and gt_pos[1] > image_height - h_tol:
                return True
            return False

        return distance < acceptable_dist
    
    def world_2_pixel(self, world_coordinate, camera_matrix):
        w = np.ones((4,), dtype=float)
        w[0:3] = world_coordinate
        xs, ys, s = camera_matrix @ w 
        # print(f'world_coordinate: {world_coordinate}, xs: {xs}, ys: {ys}, s: {s}')
        x = xs / s
        y = ys / s 
        return np.round(x).astype(int), np.round(y).astype(int)
    
    def calculate_img_reward(self, perc):        
        return (2.0/(1+np.exp(-perc*10.0))) - 1.0 
    
    def create_points(self, xmin, xmax, ymin, ymax, zmin, zmax):
        if np.isclose(zmin, zmax):
            x_coords = np.linspace(xmin, xmax, N, dtype=np.float64)
            y_coords = np.linspace(ymin, ymax, N, dtype=np.float64)
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            z_grid = np.full_like(x_grid, zmin, dtype=np.float64)
            return np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
        p1 = np.array([xmin, ymin, zmin], dtype=np.float64)
        p2 = np.array([xmax, ymax, zmin], dtype=np.float64)
        line_points = np.linspace(p1, p2, N, dtype=np.float64)
        z_values = np.linspace(zmin, zmax, N, dtype=np.float64)
        x = np.repeat(line_points[:, 0], N).astype(np.float64)
        y = np.repeat(line_points[:, 1], N).astype(np.float64)
        z = np.tile(z_values, N).astype(np.float64)
        return np.column_stack((x, y, z))


    def create_all_points(self):
        x_intervals = self.x_intervals
        y_intervals = self.y_intervals
        z_intervals = self.z_intervals
        pts = [np.array(self.create_points(xmin, xmax, ymin, ymax, zmin, zmax), dtype=np.float64) for (xmin, xmax), (ymin, ymax), (zmin, zmax) in zip(x_intervals, y_intervals, z_intervals)]
        for i in range(4):
            franka_body_sid = self.sim.model.site_name2id(f'bsite{i+1}')
            franka_site_pos = np.array(self.sim.data.site_xpos[franka_body_sid], dtype=np.float64)
            pts.append(franka_site_pos)
        self.excluded_points = np.vstack(pts).astype(np.float64)
        print('excluded_points shape: ', self.excluded_points.shape)

        
    # def world_2_pixel_vec(self, world_coordinates, camera_matrix):
    #     ones = np.ones((world_coordinates.shape[0], 1))
    #     homo_coords = np.hstack((world_coordinates, ones))
    #     proj = homo_coords @ camera_matrix.T
    #     xs, ys, s = proj[:, 0], proj[:, 1], proj[:, 2]
        
    #     s_abs = np.where(s != 0, np.abs(s), 1e-12)
    #     x_pixels = xs / s_abs
    #     y_pixels = ys / s_abs
    
    #     # with np.errstate(divide='ignore', invalid='ignore'):
    #     #     x_pixels = np.where(s != 0, xs / s, -2000)
    #     #     y_pixels = np.where(s != 0, ys / s, -2000)
    #     return np.round(x_pixels).astype(int), np.round(y_pixels).astype(int)
    
    def world_2_pixel_vec(self, world_coordinates, camera_matrix):
        world_coordinates = np.atleast_2d(np.asarray(world_coordinates, dtype=np.float64))
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        ones = np.ones((world_coordinates.shape[0], 1), dtype=np.float64)
        homo_coords = np.hstack((world_coordinates, ones))
        proj = homo_coords @ camera_matrix.T
        xs, ys, s = proj[:, 0], proj[:, 1], proj[:, 2]
        x_pixels = np.where(s < 0, xs / s, -2000)
        y_pixels = np.where(s < 0, ys / s, -2000)
        return np.round(x_pixels).astype(int), np.round(y_pixels).astype(int)



    # def check_in_region(self, camera_matrix, xyxy):
    #     x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        
    #     x_all, y_all = self.world_2_pixel_vec(self.excluded_points.copy(), camera_matrix)
    #     for i in range(x_all.shape[0]):
    #         x, y = x_all[i], y_all[i]
    #         if ((x >= x1) and (x < x2) and (y >= y1) and (y < y2)):
    #             site_pos = self.sim.data.site_xpos[self.target_sid].copy()
    #             camera_matrix = self.compute_camera_matrix()
    #             cx, cy = self.world_2_pixel(site_pos.copy(), camera_matrix) 
    #             p2 = self.excluded_points[i]
    #             ax, ay = self.world_2_pixel(p2.copy(), camera_matrix) 
                
    #             print('camera matrix: ', camera_matrix)
    #             print(f'cx: {cx}, cy: {cy}', 'site pos: ', site_pos)
    #             print(f'ax: {ax}, ay: {ay}, p2: {p2}')
    #             print(f"_x: {x}, _y: {y}, excluded: {p2}")
    #             print(f"{i}, X1: {x1}, Y1: {y1}, X2: {x2}, Y2: {y2}")
    #             return True
    #     return False
    
    def check_in_region(self, camera_matrix, xyxy):
        return False # w/o curtain
        x1, y1, x2, y2 = map(int, xyxy[:4])
        x_all, y_all = self.world_2_pixel_vec(self.excluded_points.copy(), camera_matrix)
        mask = (x_all >= x1) & (x_all < x2) & (y_all >= y1) & (y_all < y2)
        if mask.any():
            return True
        return False
    
    def close(self):
        if self.inference_type == 'async':
            self.image_queue.put("close")
            self.mask_process.join()
        
    
    
