import cv2
import time
import threading
import numpy as np
import gymnasium
from gymnasium.spaces import Box
from collections import deque
import multiprocessing as mp

OB_TYPE_1 = "MASK"
OB_TYPE_2 = "OH"
OB_TYPE_3 = "MASK_OH"


def print_with_delay_thread(text, delay=0.2):
    threading.Timer(delay, print, args=(text,)).start()


class CreateReacherEnv(gymnasium.Env):
    def __init__(self, 
                 scene_path, 
                 seed=-1, 
                 min_target_size=0.35, 
                 physics_dt=0.06, 
                 rendering_dt = 0.06, 
                 headless=True, 
                 image_stack=3, 
                 image_width=80, 
                 image_height=60, 
                 ob_type=OB_TYPE_1,
                 randomize_target_pos=False):
        
        from isaacsim import SimulationApp
        self._simulation_app = SimulationApp({"headless": headless})

        from omni.isaac.core.utils.stage import open_stage
        open_stage(usd_path=scene_path)

        if seed != -1:
            self.seed(seed)
        self._min_target_size = min_target_size
        
        self.randomize_target_pos = randomize_target_pos

        self.image_width = image_width
        self.image_height = image_height

        self._channel_axis = -1
        self.ob_type = ob_type
        if self.ob_type == OB_TYPE_1 or self.ob_type == OB_TYPE_3:
            channels = 4
        else:
            channels = 3
        
        self._image_shape = (image_height, image_width, image_stack * channels)

        self._action_history = 10
        self._image_buffer = deque([], maxlen=image_stack)
        self._action_buffer = deque([], maxlen=self._action_history)

        self._proprioception_shape = (2,)
        self._v_w_low = [-1, -3]
        self._v_w_high = [1, 3]
        self._orientation_low = np.array([-1, -1])
        self._orientation_high = np.array([1, 1])
        
        self._t2 = ([], []) # PINK
        #                         0: Magenta        1: Blue          2: Red          3: Green                    
        self._lower = np.array([[130, 190, 215], [105, 190, 215],  [0, 200, 200], [50, 110, 220]])
        self._upper = np.array([[175, 255, 255], [130, 255, 255], [40, 255, 255], [85, 250, 255]])
        self._target_names = {     
            0: "Magenta",        
            1: "Blue",         
            2: "Red", 
            3: "Green"   
        }
        self._target_oh = np.array([
            [1.0, 0.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], 
            [0.0, 0.0, 0.0, 1.0], 
            ])

        self._target_no = 0   
        
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.sensor import Camera
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        from omni.isaac.core.utils.viewports import set_camera_view
        from omni.isaac.core.prims.rigid_prim import RigidPrim
        from omni.isaac.core.world import World
        from omni.isaac.core.prims import XFormPrim 
        from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
        
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=rendering_dt)

        if not headless:
            set_camera_view(
                eye=[0, 0, 3], target=[0, 0, 0.2], 
                camera_prim_path="/OmniverseKit_Persp")

        # create_p = self.get_random_poses()

        # self._create = self._world.scene.add(
        #     Articulation(prim_path="/create_3", name="create_3", 
        #     position=np.array([create_p[0], create_p[1], 0.2]),
        #     orientation=euler_angles_to_quat([0, 0, create_p[2]], degrees=True))
        # )
        
        self._create = Articulation(
            prim_path="/create_3",
            name="create_3"
        ) 
        self._world.scene.add(self._create)
        
        if self.randomize_target_pos:
            self._square_paths = ["/target_01", "/target_02", "/target_03", "/target_04"]
            self._squares = []
            for i, path in enumerate(self._square_paths):
                sq = XFormPrim(prim_path=path, name=f"target_{i+1}")
                self._world.scene.add(sq)
                self._squares.append(sq)
        
        self._controller = DifferentialController(name="simple_control", wheel_radius=0.03575, wheel_base=0.233)
        self._robot_wheels = ["left_wheel_joint", "right_wheel_joint"]

        self._camera = Camera(
            prim_path="/create_3/base_link/rsd455/RSD455/Camera_OmniVision_OV9782_Color",
            resolution=(640, 480)
        )

        self._need_reset = True
    
    def randomize_targets(self):
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        import random
        pos_slots = [
            (0.6495, random.uniform(-0.35, 0.35), 90.0),
            (random.uniform(0.1, 0.45), -0.6495, 0.0),
            (random.uniform(-0.45, -0.1), -0.6495, 0.0),
            (-0.6495, random.uniform(-0.35, 0.35), 90.0)
        ]
        random.shuffle(self._squares)
        for sq, (px, py, yaw_deg) in zip(self._squares, pos_slots):
            sq.set_world_pose(
                position=np.array([float(px), float(py), 0.13457]),
                orientation=euler_angles_to_quat([0.0, 0.0, yaw_deg], degrees=True)
            )

    
    def reset(self):
        create_p = self.get_random_poses()

        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.rotations import euler_angles_to_quat

        # self._world.scene.remove_object("create_3", True)
        # self._create = self._world.scene.add(
        #     Articulation(prim_path="/create_3", name="create_3", 
        #     position=np.array([create_p[0], create_p[1], 0.2]),
        #     orientation=euler_angles_to_quat([0, 0, create_p[2]], degrees=True))
        # ) 

        self._world.reset()
        
        self._create.set_world_pose(
            position=[create_p[0], create_p[1], 0.2],
            orientation=euler_angles_to_quat([0, 0, create_p[2]], degrees=True)
        )
        
        if self.randomize_target_pos:
            self.randomize_targets()

        self._controller.reset()
        self._camera.initialize()
        self._target_no = np.random.choice([0, 1, 2, 3])
        pr = 'Target: ' + self._target_names[self._target_no]
        print_with_delay_thread(pr)
        
        self.latest_mask = np.zeros((self.image_height, self.image_width, 1), dtype=np.uint8)

        img = None
        for i in range(5):
            self._world.step(render=True)
            img = self._camera.get_rgb()
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.get_target_size(img)
        
        if self.ob_type == OB_TYPE_1 or self.ob_type == OB_TYPE_3:
            img = np.concatenate([img, self.latest_mask], axis=-1)
        
        img = cv2.resize(img, (self.image_width, self.image_height))
        
        for _ in range(self._image_buffer.maxlen):
            self._image_buffer.append(img)
        self._latest_image = np.concatenate(self._image_buffer, 
                                            axis=self._channel_axis)
        self._latest_proprioception = np.array([0, 0], dtype=np.float32)
        
        for _ in range(self._action_buffer.maxlen):
            self._action_buffer.append([0, 0])
        
        last_actions = np.array(self._action_buffer).reshape(self._action_history * 2)
        self._need_reset = False
        
        from omni.isaac.core.utils.rotations import quat_to_euler_angles
        orientation = quat_to_euler_angles(self._create._articulation_view.get_world_poses()[1][0], True)
        orientation = orientation[:-1] / 180
        
        if self.ob_type == OB_TYPE_2 or self.ob_type == OB_TYPE_3:
            proprioception = np.concatenate((last_actions, self._target_oh[self._target_no]))
        else:
            proprioception = last_actions
        # dof_names = self._create._articulation_view._dof_names
        # print("Available DOF names:", dof_names)

        return (self._latest_image, proprioception)


    def step(self, action):
        assert not self._need_reset

        v, w = action[0], action[1] 
        
        self._action_buffer.append([v, w])

        wheel_dof_indices = [self._create.get_dof_index(
            self._robot_wheels[i]) for i in range(len(self._robot_wheels))]
        actions = self._controller.forward(command=[v, w])
        from omni.isaac.core.utils.types import ArticulationAction
        joint_actions = ArticulationAction()
        joint_actions.joint_velocities = np.zeros(self._create.num_dof)
        if actions.joint_velocities is not None:
            for j in range(len(wheel_dof_indices)):
                joint_actions.joint_velocities[wheel_dof_indices[j]] = actions.joint_velocities[j]
        self._create.apply_action(joint_actions)

        self._world.step(render=True)

        img = self._camera.get_rgb()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # cv2.imshow('w', img)
        # cv2.waitKey(60) 
        
        target_size = self.get_target_size(img)
        reward = (2.0/(1+np.exp(-target_size*10.0))) - 1.0 
        reward = reward -1
        
        if self.ob_type == OB_TYPE_1 or self.ob_type == OB_TYPE_3:
            # mask = cv2.cvtColor(self.latest_mask, cv2.COLOR_GRAY2RGB)
            # cv2.imshow("RGB + Mask", np.hstack([img, mask]))
            # cv2.waitKey(60)
            
            img = np.concatenate([img, self.latest_mask], axis=-1) 
            
            
        img = cv2.resize(img, (self.image_width, self.image_height))
                       
        self._image_buffer.append(img)
        self._latest_image = np.concatenate(self._image_buffer,
                                            axis=self._channel_axis)
        
        from omni.isaac.core.utils.rotations import quat_to_euler_angles
        orientation = quat_to_euler_angles(self._create._articulation_view.get_world_poses()[1][0], True)

        done = False

        if abs(orientation[0]) > 30 or abs(orientation[1]) > 30:
            done = True
            reward = -100
            self._need_reset = True

        if not done and target_size >= self._min_target_size:
            done = True
            reward = 5
            self._need_reset = True

        orientation = orientation[:-1] / 180
        
        last_actions = np.array(self._action_buffer).reshape(self._action_history * 2)
        
        if self.ob_type == OB_TYPE_2 or self.ob_type == OB_TYPE_3:
            proprioception = np.concatenate((last_actions, self._target_oh[self._target_no]))
        else:
            proprioception = last_actions
        
        return (self._latest_image, proprioception), reward, done, {'size': target_size}


    def get_random_poses(self):
        create_x = np.random.uniform(low=-0.45, high=0.45)
        create_y = np.random.uniform(low=-0.45, high=0.45)
        or1 = np.random.uniform(low=0, high=359)
        
        return (create_x, create_y, or1)
        

    def get_target_size(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower[self._target_no], self._upper[self._target_no])
        
        kernel = np.ones((3, 3), 'uint8')

        # mask = cv2.dilate(mask, kernel, iterations=2)
        # mask = cv2.erode(mask, kernel, iterations=2)
        
        if self.ob_type == OB_TYPE_1 or self.ob_type == OB_TYPE_3:
            self.latest_mask = np.expand_dims(mask, axis=-1)
        
        target_size = np.sum(mask/255.) / mask.size
        
        return target_size

    def close(self):
        self._simulation_app.close()
        cv2.destroyAllWindows()


    @property
    def image_space(self):
        return Box(low=0, high=255, shape=self._image_shape)

    @property
    def proprioception_space(self):
        if self.ob_type == OB_TYPE_2 or self.ob_type == OB_TYPE_3:
            low = np.array( self._v_w_low * self._action_history + [0., 0., 0., 0.])
            high = np.array( self._v_w_high * self._action_history + [1., 1., 1., 1.]) 
            return Box(low=low, high=high)
        else:
            low = np.array(self._v_w_low * self._action_history)
            high = np.array(self._v_w_high * self._action_history)
            return Box(low=low, high=high)

    @property
    def observation_space(self):
        return self.proprioception_space
    
    @property
    def action_space(self):
        return Box(low=np.array(self._v_w_low), high=np.array(self._v_w_high))

    def seed(self, seed=None):
        self.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]
    
    

    
# if __name__ == "__main__":
#     mp.set_start_method('spawn')
#     env = CreateReacherEnv('JSAC/jsac/envs/isaac_create_reacher/create_arena.usd', 
#                            headless=True,
#                            ob_type=OB_TYPE_3,
#                            image_width=640,
#                            image_height=480)
#     state = env.reset()
    
#     print(env.proprioception_space.shape)
    
#     # for i in range (15):
#     #     next_state, reward, done, info = env.step(np.array([0.5, 0])) 
#     #     print(next_state[1])
#     #     time.sleep(0.25)
        
#     env.close()