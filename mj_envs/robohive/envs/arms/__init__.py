""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from gymnasium.envs.registration import register
from robohive.envs.env_variants import register_env_variant
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))

print("RoboHive:> Registering Arms Envs")

# FRANKA REACH =======================================================================
from robohive.envs.arms.reach_base_v0 import ReachBaseV0


register(
    id='UR10eEnv-v1',
    entry_point='robohive.envs.arms.env_v1:EnvV1',
    # max_episode_steps=200, 
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_eight.xml', 
        'robot_site_name': "pinch"    
    }
)
 
register(
    id='FrankaEnv-v0',
    entry_point='robohive.envs.arms.env_v0:EnvV0',
    kwargs={
        'model_path': curr_dir+'/franka/scene_eight.xml',
        'robot_site_name': "end_effector",
    }
)
 
register(
    id='FrankaEnv-v1',
    entry_point='robohive.envs.arms.env_v1:EnvV1',
    kwargs={
        'model_path': curr_dir+'/franka/scene_eight.xml',
        'robot_site_name': "end_effector",
    }
)
