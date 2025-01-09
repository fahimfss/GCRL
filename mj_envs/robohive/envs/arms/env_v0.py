import os
import time
import copy
import math
import torch
import random
import torch.jit
import cv2 as cv
import collections
import numpy as np
from PIL import Image
import gymnasium as gym
import multiprocessing as mp
from torchvision.ops import box_convert

from robohive.envs import env_base_0
import groundingdino.datasets.transforms as T 
from robohive.utils.quat_math import mat2euler
from robohive.envs.arms.python_api_2 import BodyIdInfo, get_touching_objects, ObjLabels
from groundingdino.util.inference import load_model, predict


BOX_THRESHOLD = 0.55
TEXT_THRESHOLD = 0.25

def load_image(image_source: torch.Tensor, image_size: int):
        transform = T.Compose(
            [
                T.RandomResize(image_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(image_source, None)
        return image_transformed

def create_mask(image_source, boxes) -> np.ndarray:
        """
        This function creates a mask with white rectangles on a black background,
        where the rectangles are defined by the bounding boxes.

        Parameters:
        image_source (np.ndarray): The source image for determining the size of the mask.
        boxes (torch.Tensor): A tensor containing bounding box coordinates in cxcywh format.

        Returns:
        np.ndarray: The mask image.
        """
        # Get the dimensions of the source image
        h, w = image_source.shape

        # Scale the boxes to the image dimensions
        boxes = torch.tensor(boxes, dtype=torch.float32) * torch.Tensor([w, h, w, h])

        # Convert boxes from cxcywh to xyxy format
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Create a black mask
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (-2000, -2000)
            
        if xyxy.size != 0:
            px1, px2 = float(xyxy[0]) / w, float(xyxy[2]) / w
            py1, py2 = float(xyxy[1]) / h, float(xyxy[3]) / h
            
            if px2 - px1 > 0.8 and py2 - py1 > 0.8:
                pass
            elif px2 - px1 > 0.26 and px2 - px1 < 0.38 and py2 - py1 > 0.18 and py2 - py1 < 0.28 and py1 > 0.72 and py2 > 0.94:
                pass 
            elif px2 - px1 > 0.06 and px2 - px1 < 0.15 and py2 - py1 > 0.14 and py2 - py1 < 0.28 and py1 > 0.70 and py2 > 0.88:
                pass 
            else: 
                top_left = (int(xyxy[0]), int(xyxy[1]))
                bottom_right = (int(xyxy[2]), int(xyxy[3]))
                center = (int((top_left[0] + bottom_right[0]) / 2), int((top_left[1] + bottom_right[1]) / 2))
                cv.rectangle(mask, top_left, bottom_right, (255), thickness=-1)  # Fill the rectangle
                white_pixels = np.argwhere(mask == 255)
            
                # Calculate the mean of each column (x, y coordinates)
                centroid = np.mean(white_pixels, axis=0).astype(int)  # Returns (y, x)

                # Convert from (row, col) to (x, y)
                centroid = (centroid[1], centroid[0])

        return mask, center
    
def async_g_dino(img_shape, mem_name, image_queue, mask_queue):
    model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", 
                    "../GroundingDINO/asset/groundingdino_swinb_cogcoor.pth")
    
    # model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
    #                    "../GroundingDINO/asset/groundingdino_swint_ogc.pth")
    count = 0
    
    img_shm = mp.shared_memory.SharedMemory(name=mem_name) 
    img = np.ndarray(img_shape, dtype=np.uint8, buffer=img_shm.buf)
    h, w, c = img_shape 
    
    while True:
        data = image_queue.get() 
        if data == 'close':
            img_shm.close()
            return 
        target_name = data  
        
        pil_image = Image.fromarray(img.copy())   
        t1 = time.time()
        boxes, logits, phrases = predict(
            model=model,
            image=load_image(pil_image, [w, h]),
            caption=target_name,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        ) 
        t2 = time.time() 
        count += 1 
        if logits.nelement() > 0:
            _, indices = torch.max(logits, dim = 0)
            boxes = boxes.numpy()
            boxes = boxes[indices]
        mask_queue.put((boxes, t2 - t1, count))

def is_gdino_accurate(gt_pos, gdino_pos, dist, image_width, image_height):
    mlt = (-0.5 * dist) + 1.5 
    acceptable_dist = (image_width / 7.5) * mlt 
    if isinstance(acceptable_dist, np.ndarray):
        acceptable_dist = acceptable_dist[0]
    distance = math.sqrt((gt_pos[0] - gdino_pos[0])**2 + (gt_pos[1] - gdino_pos[1])**2)
    
    if gdino_pos[0] == -2000 and gdino_pos[1] == -2000:
        w_tol = image_width * 0.025
        h_tol = image_height * 0.025
        if gt_pos[0] < w_tol and gt_pos[0] > image_width - w_tol and gt_pos[1] < h_tol and gt_pos[1] > image_height - h_tol:
            # print("\t\t", gt_pos, gdino_pos, distance, acceptable_dist, "Out of bounds")
            return True
        return False

    # print("\t\t", gt_pos, gdino_pos, distance, acceptable_dist)
    return distance < acceptable_dist

class EnvV0(env_base_0.MujocoEnv):
    DEFAULT_OBS_KEYS = ['qp_robot', 'prev_action']
    
    DEFAULT_PROPRIO_KEYS = ['qp_robot', 'prev_action']
    
    BOX_THRESHOLD = 0.4
    TEXT_THRESHOLD = 0.25

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
               frame_skip = 25, 
               env_mode = "train",          # "train", "eval_ofd", "eval", "inference_1", "inference_3"
               reward_mode = "mask_size",   # "distance", "mask_size"
               mask_type = "ground_truth",  # "ground_truth", "gdino_sync", "gdino_async"
               mask_delay_type = "none",    # "none", "n_step", "sequential"
               mask_delay_steps = 2,
               obs_keys=DEFAULT_OBS_KEYS,
               proprio_keys=DEFAULT_PROPRIO_KEYS,
               **kwargs,
        ):

        # ids 
        self.grasp_sid = self.sim.model.site_name2id(robot_site_name) #robot part name
        self.center_obj_range = np.array([[-0.16, 0.16], [0.25, 0.45]])
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height  
        self.fixed_positions = None
        self.cam_init = True
        self._setup_camera() 
        self.current_image = np.ones((image_height, image_width, 4), dtype=np.uint8) 
        self.mask_size = 0  
        self.single_touch = 0
        
        self.target_x, self.target_y = 0, 0
        self.target_r = 0
        self.camera_matrix = None
        self.current_mask = None
        self.gdino_time = 0
        self.gdino_step = 0
        self.gdino_error = 0
        self.gdino_num_accurate = 0
        self.gdino_accuracy = 0
        self.gs = 0
        self.distance = 1.0
        self.TM = time.time()
        self.prev_action = np.array([0] * self.sim.model.nu)
        self.action_scale = 1.0
        self.roi_no = 0
        self.rois = [(.3, .7, .4, .75), (.25, .75, .35, .8)]
        
        self.env_mode = env_mode
        self.reward_mode = reward_mode
        self.mask_type = mask_type
        self.mask_delay_type = mask_delay_type
        self.mask_delay_steps = mask_delay_steps
        
        if reward_mode == "distance":
            weighted_reward_keys = {
                "distance": -1.0, 
                "contact": 0.,
                'penalty': 0.1,
                'mask_size': 0.,
                "done": 5.,
            }
        else:
            weighted_reward_keys = {
                "distance": 0., 
                "contact": 0.,
                'penalty': 1.,
                'mask_size': 0.9,
                "done": 5.,
            }
            
        if self.mask_type == "gdino_sync":
            # self.mask_model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
            #                              "../GroundingDINO/asset/groundingdino_swint_ogc.pth")
            
            self.mask_model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", 
                               "../GroundingDINO/asset/groundingdino_swinb_cogcoor.pth")
            
        elif self.mask_type == "gdino_async":
            self.image_queue = mp.Queue()
            self.mask_queue = mp.Queue()
            
            img_shape = (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)
            original_array = np.zeros(img_shape, dtype=np.uint8)
            self.img_shm = mp.shared_memory.SharedMemory(create=True, size=original_array.nbytes)
            self.img_arr = np.ndarray(img_shape, dtype=np.uint8, buffer=self.img_shm .buf)
            
            self.mask_process = mp.Process(target=async_g_dino, args=(img_shape, self.img_shm.name, self.image_queue, self.mask_queue))
            # async_g_dino(img_shape, mem_name, image_queue, mask_queue):
            self.mask_process.start()
            self.images_sent = 0
            self.masks_recieved = 0
                
        if self.mask_delay_type == "n_step":
            self.mask_step = -1
        
        self.target_name = "apple"
        
        self.TS = ['object_1',  'object_2',    'object_3',          'object_4',        'object_5',        'object_6',        'object_7',     'object_8']
        self.TN = ['red apple', 'green block', 'round donut', 'glass flask jar', 'yellow duck toy', 'yellow banana',   'purple clock', 'cup'     ]
        # self.TN = ['apple',    'green block', 'donut',    'beaker',   'rubber duck', 'banana',   'alarm clock', 'cup'     ]
        self.target_sid = self.sim.model.site_name2id(self.TS[0]) 
        self.r = 2
        self.target_site_name = self.TS[0]
        
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
        mask_size_reward = np.array([self.calculate_img_reward(self.mask_size)])
        contact = np.array([np.sum(obs_dict["touching_body"][0][0][:2])])

        if contact == 1:
            self.single_touch += 1
            if self.single_touch == 1:
                print('First touch!')
        elif contact == 2:
            self.single_touch += 1
            print('Second touch!') 
             
        rwd_dict = collections.OrderedDict((
            ('distance',  self.distance),
            ('contact', contact),
            ('penalty', np.array([-1])),  
            ('mask_size',  mask_size_reward),
            ('done', contact == 2),  
        )) 
        
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        rwd_dict['done'] = False
         
        # if self.env_mode == "train":
        #     rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        # else:
        #     rwd_dict['dense'] = 1.0 if contact == 2 else 0
        #     rwd_dict['done'] = contact == 2
        
        return rwd_dict
    
    def place_object(self, obj_name, reset_qpos, x_pos=None, y_pos=None, drop=False):
        objec_bid = self.sim.model.body_name2id(obj_name)  # get body ID using object name
        object_jnt_adr = self.sim.model.body_jntadr[objec_bid]
        object_qpos_adr = self.sim.model.jnt_qposadr[object_jnt_adr]
        if drop: 
            reset_qpos[object_qpos_adr + 2] = 0.4
        else:
            reset_qpos[object_qpos_adr] = x_pos
            reset_qpos[object_qpos_adr + 1] = y_pos

        if obj_name == 'object_4':    
            objec_bid = self.sim.model.body_name2id('base_rbf')
            object_jnt_adr = self.sim.model.body_jntadr[objec_bid]
            object_qpos_adr = self.sim.model.jnt_qposadr[object_jnt_adr]
            if drop: 
                reset_qpos[object_qpos_adr + 2] = 0.4
            else:
                reset_qpos[object_qpos_adr] = x_pos
                reset_qpos[object_qpos_adr + 1] = y_pos
    
    def reset(self, reset_qpos=None, **kwargs): 
        # print("-->", self.gdino_num_accurate, self.gs, self.gdino_accuracy)
        self.prev_action = np.array([0] * self.sim.model.nu)
        self.current_mask = None
        self.gdino_error = 0
        self.gdino_num_accurate = 0
        self.gdino_accuracy = 0
        self.distance = 1.0
        self.gs = 0
        self.single_touch = 0
        
        if 'action_scale' in kwargs:
            self.action_scale = kwargs['action_scale']
            kwargs.pop('action_scale')
        if 'roi' in kwargs:
            self.roi_no = kwargs['roi']
            kwargs.pop('roi')

        if self.mask_delay_type == "n_step":
            self.mask_step = -1
        
        if self.mask_type == "gdino_async":
            while(self.images_sent > self.masks_recieved):
                self.mask_queue.get()
                self.masks_recieved += 1
            self.images_sent = 0
            self.masks_recieved = 0
         
        if self.env_mode == "train":
            number = np.random.randint(0, 5)
        elif self.env_mode == "eval_ofd":
            number = np.random.randint(5, 8)
        elif self.env_mode == "eval":
            number = np.random.randint(0, 5)
        else:
            number = self.target_obj_num 
        
        reset_qpos = self.sim.model.key_qpos[0].copy()
        
        self.target_name = self.TN[number] 
        self.target_site_name = self.TS[number] 
        # print("target:", self.target_name)
        self.target_sid = self.sim.model.site_name2id(self.target_site_name) 
 
        if self.env_mode == "inference_1" or self.env_mode == "inference_3":
            if self.env_mode == "inference_3":
                other_indices = random.sample([i for i in range(8) if i != number], 2)
                other_site_names = [self.TS[ind] for ind in other_indices]
            else:
                other_site_names = []
            
            for obj_name in self.TS:
                if obj_name == self.target_site_name:
                    x_pos = random.uniform(-0.03, 0.03)
                    y_pos = 0.35 + random.uniform(-0.02, 0.02)
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
            if self.env_mode == "eval_ofd":
                site_names = random.sample(self.TS[3:8], 5)
                for obj_name in self.TS[0:3]:
                    self.place_object(obj_name, reset_qpos, drop=True)
            else:
                site_names = random.sample(self.TS[0:5], 5)
                for obj_name in self.TS[5:8]:
                    self.place_object(obj_name, reset_qpos, drop=True)
            
            center_obj_x_pos = random.uniform(self.center_obj_range[0][0], 
                                              self.center_obj_range[0][1])
            
            center_obj_y_pos = random.uniform(self.center_obj_range[1][0], 
                                              self.center_obj_range[1][1])

            for index, obj_name in enumerate(site_names):
                x_pos = center_obj_x_pos + (((index - 2) * 0.165) +  random.uniform(-0.02, 0.02))
                y_pos = center_obj_y_pos + random.uniform(-0.05, 0.05)
                self.place_object(obj_name, reset_qpos, x_pos, y_pos)
         
        obs = super().reset(reset_qpos = reset_qpos, reset_qvel = None, **kwargs)
        
        site_pos = self.sim.data.site_xpos[self.target_sid]
        camera_matrix = self.compute_camera_matrix()
        self.target_x, self.target_y  = self.world_2_pixel(site_pos, camera_matrix) 
        site_pos[0] += 0.04
        rx, ry  = self.world_2_pixel(site_pos, camera_matrix) 
        self.r = math.sqrt((rx - self.target_x) ** 2 + (ry - self.target_y) ** 2)
        
        self.final_image = self.current_image
        
        self.TM = time.time()
        return {'image': self.final_image, 'vector': obs}
    

    def get_observation(self):
        """
        Uses the controllers get_image_data method to return an top-down image (as a np-array).

        Args:
            show: If True, displays the observation in a cv2 window.
        """

        rgb = self.get_image_data()
        
        site_pos = self.sim.data.site_xpos[self.target_sid].copy()
        camera_matrix = self.compute_camera_matrix()
        self.target_x, self.target_y = self.world_2_pixel(site_pos, camera_matrix) 
        site_pos[0] += 0.04
        rx, ry  = self.world_2_pixel(site_pos, camera_matrix) 
        self.r = math.sqrt((rx - self.target_x) ** 2 + (ry - self.target_y) ** 2)

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
        a = a * self.action_scale
        self.prev_action = a
 
        if self.single_touch >= 1000:
            print('hard-coded')
            self.fixed_positions = self.sim.data.qpos[:self.sim.model.nu].copy()
            self.fixed_positions[-1] = 1
            a[-1] = 1

            self.last_ctrl = self.robot.step(ctrl_desired=a,
                                        last_qpos = self.fixed_positions,
                                        dt = self.dt,
                                        render_cbk=self.mj_render if self.mujoco_render_frames else None)
        else:
            a = np.clip(a, self.action_space.low, self.action_space.high)
            self.fixed_positions = None

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
        rgb = copy.deepcopy(
            self.sim.renderer.render_offscreen(width=self.IMAGE_WIDTH, 
                                               height=self.IMAGE_HEIGHT, 
                                               camera_id=camera, depth = False)) 

        rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
        
        if self.mask_type == "ground_truth":
            mask = np.zeros((self.IMAGE_HEIGHT,  self.IMAGE_WIDTH), dtype=np.uint8)
            x, y = int(self.target_x), int(self.target_y)
            half_side = int(max(self.r, 2))
            cv.rectangle(mask, (x - half_side, y - half_side), (x + half_side, y + half_side), 255, thickness=-1)

            if self.mask_delay_type == "none":
                self.current_mask = mask
            elif self.mask_delay_type == "n_step":
                if self.mask_step == -1:
                    self.current_mask = mask.copy()
                    self.saved_mask = mask.copy()
                    self.mask_step = 1
                else:
                    if self.mask_step == 0:
                        self.current_mask = self.saved_mask
                        self.saved_mask = mask.copy()
                        self.mask_step = self.mask_delay_steps
                self.mask_step -= 1
        elif self.mask_type == "gdino_sync":
            pil_image = Image.fromarray(rgb)
            t1 = time.time()
            boxes, logits, phrases = predict(
                model=self.mask_model,
                image=load_image(pil_image, [self.IMAGE_WIDTH, self.IMAGE_HEIGHT]),
                caption=self.target_name,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )
            t2 = time.time() 
            self.gdino_step += 1
            self.gdino_time = t2 - t1
            if logits.nelement() > 0:
                _, indices = torch.max(logits, dim = 0)
                boxes = boxes.numpy()
                boxes = boxes[indices] 
            mask = np.zeros((self.IMAGE_HEIGHT,  self.IMAGE_WIDTH), dtype=np.uint8) 
            self.current_mask, self.gdino_center = create_mask(mask, boxes=boxes)
            
            gt_center = int(self.target_x), int(self.target_y)
            if is_gdino_accurate(gt_center, self.gdino_center, self.distance, self.IMAGE_WIDTH, self.IMAGE_HEIGHT):
                self.gdino_num_accurate += 1
            self.gs += 1
            self.gdino_accuracy = float(self.gdino_num_accurate) / self.gs
            
            # print(self.gdino_num_accurate, self.gs, self.gdino_accuracy, self.gdino_time)

        elif self.mask_type == "gdino_async":
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
            if is_gdino_accurate(gt_center, self.gdino_center, self.distance, self.IMAGE_WIDTH, self.IMAGE_HEIGHT):
                self.gdino_num_accurate += 1
            self.gs += 1
            self.gdino_accuracy = float(self.gdino_num_accurate) / self.gs
            
            # print(self.gdino_num_accurate, self.gs, self.gdino_accuracy, self.gdino_time, "  target: ", gt_center)
            
        #define the grasping rectangle
        r1, r2, r3, r4 = self.rois[self.roi_no]
        x1, x2 = int(self.IMAGE_WIDTH * r1), int(self.IMAGE_WIDTH * r2)
        y1, y2 = int(self.IMAGE_HEIGHT * r3), int(self.IMAGE_HEIGHT * r4)
        
        # cv.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        roi = self.current_mask[y1:y2, x1:x2]
        white_pixels = float(np.sum(roi == 255))
        total_pixels = float(roi.size)
        self.mask_size = (white_pixels / total_pixels)  

        self.current_image = np.concatenate((rgb, np.expand_dims(self.current_mask, axis=-1)), axis=2)
         
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
        return (2.0/(1+np.exp(-perc*8.0))) - 1.0 
    
    def close(self):
        self.image_queue.put("close")
        self.mask_process.join()
    
    