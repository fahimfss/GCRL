import time
import copy
import math

import random
import cv2 as cv
import collections
import numpy as np
import gymnasium as gym
from robohive.envs import env_base_0

from robohive.utils.quat_math import mat2euler
from robohive.envs.arms.python_api_2 import BodyIdInfo, get_touching_objects, ObjLabels

GOALTYPE_MASK = "G1_Mask"
GOALTYPE_ONE_HOT = "G2_OH"
GOALTYPE_3D = "G3_3d"
GOALTYPE_CLIP = "G4_Clip"
GOALTYPE_TARGET_STATE = "G5_TS"
MASK_SIZE_LIMIT = 30
MASK_SIZE_LIMIT_DIST = 30
DISTANCE_THRESHOLD = 0.1

class EnvV1(env_base_0.MujocoEnv):
    
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
               image_width=640,
               image_height=480,
               frame_skip = 20, 
               env_mode = "train",          # "train", "eval_ofd", "eval", "inference_1", "inference_3"
               reward_mode = "distance",    # "distance", "mask_size"
               goal_type = GOALTYPE_MASK,     
               ofd_index=0,
               **kwargs,
        ):

        self.grasp_sid = self.sim.model.site_name2id(robot_site_name) #robot part name
        self.center_obj_range = np.array([[-0.15, 0.15], [0.29, 0.41]])
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height  
        self.cam_init = True
        self._setup_camera() 
        self.goal_type = goal_type
        self.target_x, self.target_y = 0, 0
        self.target_r = 0
        self.camera_matrix = None
        self.current_mask = None
        self.distance = 1.0
        self.prev_action = np.array([0] * self.sim.model.nu)
        self.env_mode = env_mode
        self.reward_mode = reward_mode 
        self.target_name = "red apple"
        self.mask_size = 0  
        self.mask_size_counter = 0
        self._min_dist = 10000
 
        if reward_mode == "distance":
            weighted_reward_keys = {
                "distance": -1.0, 
                "contact": 0.,
                'penalty': 0.,
                'mask_size': 0.,
                "done": 5.,
            }
        else:
            weighted_reward_keys = {
                "distance": 0., 
                "contact": 0.,
                'penalty': 1.,
                'mask_size': 1.,
                "done": 5.,
            }
        
        self.objects = {
            #site_name: object name, (width_scale, height_scale), (width_offset, height_offset)
            'object_1': ('red apple', (1.25, 1.35), (0, -0.03)),
            'object_2': ('green block', (1.1, 1.1), (0, -0.02)),
            'object_3': ('chocolate donut', (1.1, 1.1), (0, 0)),
            'object_4': ('round bottomed flask', (1.1, 1.5), (0, -0.03)),
            'object_5': ('yellow toy duck', (1.4, 1.3), (0, -0.02)),
            'object_6': ('yellow banana', (1.3, 1.7), (-0.02, 0)),
            'object_7': ('purple alarm clock', (1.4, 1.4), (0, -0.03)),
            'object_8': ('cup', (1.4, 1.5), (0, -0.04)),
            'object_9': ('blue water bottle', (1.1, 1.4), (0, 0)),
            'object_10': ('light bulb', (1.4, 1.1), (0, 0)),
            'object_11': ('wine glass', (1.1, 1.35), (0, 0)),
            'object_12': ('copper bowl', (1.35, 1.2), (0, -0.02)),
            'object_13': ('silver headphone', (1.2, 1.2), (0, 0)),
            'object_14': ('hammer', (1.1, 1.4), (-0.02, -0.02)),
            'object_15': ('digital camera', (1.45, 1.1), (0, 0)),
            'object_16': ('blue stapler', (1.45, 1), (0, 0)),
            'object_17': ('white egg', (1, 1), (0, 0)),
            'object_18': ('green toy train', (1.4, 1), (0, 0)),
            'object_19': ('teapot', (1.4, 1.3), (0, 0)),
            'object_20': ('red eyeglasses', (1.4, 1.2), (0, 0))
        }
        
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

        self.TS = list(self.objects.keys())
        self.TN = []
        self.mask_scale = []
        self.mask_offset = []
        for item in self.objects.values():
            self.TN.append(item[0])
            self.mask_scale.append(item[1])
            self.mask_offset.append(item[2]) 
        
        self.target_sid = self.sim.model.site_name2id(self.TS[0]) 
        self.r = 0
        self.target_site_name = self.TS[0]
        self.current_mask_scale = (1, 1)
        self.current_mask_offset = (0, 0)
        
        if 'target_obj_num' in kwargs:
            self.target_obj_num = kwargs['target_obj_num'] 
            kwargs.pop('target_obj_num')

        if self.goal_type == GOALTYPE_MASK:
            self.current_image = np.ones((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 4), dtype=np.uint8) 
            self.obs_keys = ['qp_robot', 'prev_action']
            self.proprio_keys = self.obs_keys.copy()
        elif self.goal_type == GOALTYPE_ONE_HOT:
            self.oh_vec = np.array([0.0] * 20)
            self.current_image = np.ones((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3), dtype=np.uint8) 
            self.obs_keys = ['qp_robot', 'prev_action', 'one_hot']
            self.proprio_keys = self.obs_keys.copy()
        elif self.goal_type == GOALTYPE_3D:
            self.target_3d_pos = np.array([0.0] * 3)
            self.current_image = np.ones((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3), dtype=np.uint8) 
            self.obs_keys = ['qp_robot', 'prev_action', '3d_pos']
            self.proprio_keys = self.obs_keys.copy()
        elif self.goal_type == GOALTYPE_CLIP:
            self.clip_embeddings = np.load('/gpfs/home/wanghuiy/RLC/mj_envs/robohive/envs/arms/gt_targets/embeddings.npy')
            self.current_clip_embedding = self.clip_embeddings[0].copy()
            self.current_image = np.ones((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3), dtype=np.uint8) 
            self.obs_keys = ['qp_robot', 'prev_action', 'clip_embedding']
            self.proprio_keys = self.obs_keys.copy()
        elif self.goal_type == GOALTYPE_TARGET_STATE:
            paths = [f'../mj_envs/robohive/envs/arms/gt_targets/{i}.png' for i in range(20)]
            imgs = [cv.imread(p) for p in paths]
            self.target_state_images = np.array(imgs)    
            self.current_image = np.ones((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 6), dtype=np.uint8) 
            self.obs_keys = ['qp_robot', 'prev_action']
            self.proprio_keys = self.obs_keys.copy()
            self.current_target_state_image = self.target_state_images[0]
        
        super()._setup(obs_keys=self.obs_keys,
                       proprio_keys=self.proprio_keys,
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
        
        if self.goal_type == GOALTYPE_ONE_HOT:
            obs_dict['one_hot'] = self.oh_vec
        elif self.goal_type == GOALTYPE_3D:
            self.target_3d_pos = sim.data.site_xpos[self.target_sid]
            obs_dict['3d_pos'] = self.target_3d_pos
        elif self.goal_type == GOALTYPE_CLIP: 
            obs_dict['clip_embedding'] = self.current_clip_embedding
 
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
        if self.reward_mode == 'mask_size': 
            if mask_size >= MASK_SIZE_LIMIT:
                self.mask_size_counter += 1
            done_1 = np.array([self.mask_size_counter]) == 5
            done_2 = self.mask_size_counter == 5
        else:
            if self.distance < DISTANCE_THRESHOLD and mask_size >= MASK_SIZE_LIMIT_DIST:
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
        self.distance = 1.0

        self.mask_size_counter = 0 
        
        ofd_items = self.ofd[self.ofd_index]
        train_items = list(set(range(20)) - set(ofd_items))
        
        if self.env_mode == "train" or self.env_mode == "eval":
            number = random.choice(train_items)
        elif self.env_mode == "eval_ofd":
            number = random.choice(ofd_items) 
        else:
            number = self.target_obj_num 
             
        reset_qpos = self.sim.model.key_qpos[0].copy()
        
        self.target_name = self.TN[number] 
        self.target_site_name = self.TS[number] 
        self.current_mask_scale = self.mask_scale[number]
        self.current_mask_offset = self.mask_offset[number]
        
        if self.goal_type == GOALTYPE_ONE_HOT:
            self.oh_vec = np.array([0.0] * 20)
            self.oh_vec[number] = 1.0
        if self.goal_type == GOALTYPE_TARGET_STATE:
            self.current_target_state_image = self.target_state_images[number]
        
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
                    x_pos = -0.02
                    y_pos = 0.31

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
        
        site_pos = self.sim.data.site_xpos[self.target_sid].copy()
        camera_matrix = self.compute_camera_matrix()
        self.target_x, self.target_y = self.world_2_pixel(site_pos, camera_matrix) 
        site_pos[0] += 0.04
        rx, ry  = self.world_2_pixel(site_pos, camera_matrix) 
        try:
            self.r = math.sqrt((rx - self.target_x) ** 2 + (ry - self.target_y) ** 2)
        except:
            self.r = 0

        observation = {}
        observation["rgb"] = rgb
        
        return observation
    
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
        rgb = copy.deepcopy(
            self.sim.renderer.render_offscreen(width=self.IMAGE_WIDTH, 
                                               height=self.IMAGE_HEIGHT, 
                                               camera_id=camera, depth = False)) 

        rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
         
        mask = np.zeros((self.IMAGE_HEIGHT,  self.IMAGE_WIDTH), dtype=np.uint8)
        x, y = int(self.target_x), int(self.target_y)
        o1, o2 = self.current_mask_offset
        o1, o2 = int(o1 * self.IMAGE_WIDTH), int(o2 * self.IMAGE_HEIGHT)
        x, y = x + o1, y + o2
        
        half_side = int(max(self.r, 2))
        if half_side < 1000:
            hs1 = int(half_side * self.current_mask_scale[0]) 
            hs2 = int(half_side * self.current_mask_scale[1])
            cv.rectangle(mask, (x - hs1, y - hs2), (x + hs1, y + hs2), 255, thickness=-1)

        self.current_mask = mask

        # x1, x2 = int(self.IMAGE_WIDTH * 0.20), int(self.IMAGE_WIDTH * 0.80)
        # y1, y2 = int(self.IMAGE_HEIGHT * 0.30), int(self.IMAGE_HEIGHT * 0.80)
        
        x1, x2 = 0, self.IMAGE_WIDTH
        y1, y2 = 0, self.IMAGE_HEIGHT
        
        roi = self.current_mask[y1:y2, x1:x2]
        white_pixels = float(np.sum(roi == 255))
        total_pixels = float(roi.size)
        self.mask_size = (white_pixels / total_pixels)  

        if self.goal_type == GOALTYPE_MASK:
            self.current_image = np.dstack((rgb, self.current_mask))
        elif self.goal_type == GOALTYPE_TARGET_STATE:
            self.current_image = np.dstack((rgb, self.current_target_state_image))
        else:
            self.current_image = rgb
        
        return np.array(np.fliplr(np.flipud(rgb)))

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
        """Returns the 3x4 camera matrix."""
        # If the camera is a 'free' camera, we get its position and orientation
        # from the scene data structure. It is a stereo camera, so we average over
        # the left and right channels. Note: we call `self.update()` in order to
        # ensure that the contents of `scene.camera` are correct.

        pos = self.sim.data.cam_xpos[self.sim.model.camera_name2id(camera)]
        rot_mat = self.sim.data.cam_xmat[self.sim.model.camera_name2id(camera)].reshape(3, 3)
        camera_id = self.sim.model.camera_name2id(camera)
        fov = self.sim.model.cam_fovy[camera_id]

        # Translation matrix (4x4).
        translation = np.eye(4)
        translation[0:3, 3] = -pos

        # Rotation matrix (4x4).
        rotation = np.eye(4)
        rotation[0:3, 0:3] = rot_mat.T

        # Focal transformation matrix (3x4).
        focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * self.IMAGE_HEIGHT / 2.0
        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

        # Image matrix (3x3).
        image = np.eye(3)
        image[0, 2] = (self.IMAGE_WIDTH - 1) / 2.0
        image[1, 2] = (self.IMAGE_HEIGHT - 1) / 2.0
        return image @ focal @ rotation @ translation
    
    def world_2_pixel(self, world_coordinate, camera_matrix):
        """
        Takes a XYZ world position and transforms it into pixel coordinates.
        Mainly implemented for testing the correctness of the camera matrix, focal length etc.

        Args:
            world_coordinate: XYZ world coordinate to be transformed into pixel space.
            width: Width of the image (pixel).
            height: Height of the image (pixel).
            camera: Name of camera used to obtain the image.
        """
        
        w = np.ones((4,), dtype=float)
        w[0:3] = world_coordinate
        xs, ys, s = camera_matrix @ w 
        x = xs / s
        y = ys / s 
        return np.round(x).astype(int), np.round(y).astype(int)
    
    def calculate_img_reward(self, perc):        
        return (2.0/(1+np.exp(-perc*10.0))) - 1.0 
    
    def close(self):
        pass
    
    