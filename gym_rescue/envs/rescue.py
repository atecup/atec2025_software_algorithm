# from gym_unrealcv import settings
import time
from gym_rescue.envs.base_env import UnrealCv_base
import numpy as np
import random
'''
Tasks: The task is to make the agents find the target(injured person) and rescue him. 
The agents are allowed to communicate with others.
The agents observe the environment with their own camera.
The agents are rewarded based on the distance and orientation to the target.
The episode ends when the agents reach the target(injured agent) or the maximum steps are reached.
'''

class Rescue(UnrealCv_base):
    def __init__(self,
                 env_file,  # the setting file to define the task
                 task_file=None,  # the file to define the task TODO: use this file to config task specific parameters
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 resolution=(160, 160),
                 reset_type=0
                 ):
        super(Rescue, self).__init__(setting_file=env_file,  # the setting file to define the task
                                     action_type=action_type,  # 'discrete', 'continuous'
                                     observation_type=observation_type,  # 'color', 'depth', 'rgbd', 'Gray'
                                     resolution=resolution,
                                     reset_type=reset_type)
        self.count_reach = 0
        self.max_reach_steps = 5
        self.distance_threshold = 120
        self.agents_category = ['player']
        self.injured_agent = self.env_configs['injured_player'][0]
        self.stretcher=self.env_configs['stretcher'][0]
        self.ambulance = self.env_configs['ambulance'][0]

        # self.rescue_pose = self.env_configs['stretcher_loc'][0]
        # level_list = [l for l in self.env_configs.keys() if l.startswith('level')]
        # assert len(level_list)>0
        # self.injured_player_pose=self.env_configs[level_list[0]]['injured_player_loc'][reset_type]
        # self.rescue_pose = self.env_configs[level_list[0]]['stretcher_loc'][reset_type]
        # self.agent_pose = self.env_configs[level_list[0]]['agent_loc'][reset_type]
        # self.ambulance_pose= self.env_configs[level_list[0]]['ambulance_loc'][reset_type]
        self.injured_player_pose = None
        self.rescue_pose = None
        self.agent_pose = None
        self.ambulance_pose = None

    def step(self, action):
        obs, reward, termination, truncation, info = super(Rescue, self).step(action)
        # compute the useful metrics for rewards and done condition
        current_pose = [self.unrealcv.get_cam_pose(self.cam_list[self.protagonist_id])]
        metrics = self.rescue_metrics(current_pose, self.target_pose)
        info['picked']=metrics['picked']
        if info['picked']:
            self.target_pose = self.rescue_pose
            if self.first_picked is False:  # the first time the agent pick the injured agent
                reward = 0.5
                self.first_picked = True

        # done condition
        current_injured_player_pose = self.unrealcv.get_obj_location(self.injured_agent)+self.unrealcv.get_obj_rotation(self.injured_agent)
        _, dis_tmp, dire_tmp=self.get_relative(current_injured_player_pose, self.rescue_pose)

        if not metrics['picked'] and dis_tmp<200: # the agent place the injured agent on the stretcher
            info['termination'] = True
            info['truncation'] = False
            reward = 0.5
            # if current_injured_player_pose[2]-self.rescue_pose[2] > 120:#the injured agent is on the stretcher
            #     self.count_reach+=1
            #     if self.count_reach > self.max_reach_steps:
            #         info['termination']=True
            #         info['truncation']=False
        else:
            self.count_reach= 0
        info['Reward'] = reward

        return obs, reward, info['termination'],info['truncation'], info

    def reset(self):
        # initialize the environment
        states, info = super(Rescue, self).reset()
        # super(Rescue, self).random_app()
        # for i in range(15):
        #     self.unrealcv.set_appearance(self.injured_agent, i)
        #     self.unrealcv.get_obj_location(self.injured_agent)
        #     time.sleep(2)
        # self.unrealcv.set_appearance(self.player_list[self.protagonist_id], 10)
        self.target_pose = self.injured_player_pose
        self.count_reach = 0
        self.first_picked = False

        #NPC will randomly walking in the environment
        for i in range(len(self.player_list)):
            if i !=self.protagonist_id:
                self.unrealcv.nav_random(self.player_list[i],1000,1)
        self.player_list = [self.player_list[self.protagonist_id]]

        #initialize start location
        self.unrealcv.set_obj_rotation(self.player_list[self.protagonist_id], self.unrealcv.get_obj_rotation(self.injured_agent))
        self.unrealcv.set_obj_location(self.player_list[self.protagonist_id], self.unrealcv.get_obj_location(self.injured_agent))
        self.unrealcv.carry_body(self.player_list[self.protagonist_id])
        time.sleep(1)
        self.unrealcv.drop_body(self.player_list[self.protagonist_id])
        time.sleep(1)
        self.unrealcv.set_obj_rotation(self.player_list[self.protagonist_id], self.agent_pose[3:])
        self.unrealcv.set_obj_location(self.player_list[self.protagonist_id], self.agent_pose[:3])
        self.unrealcv.set_cam_fov(self.cam_list[self.protagonist_id],110)
        time.sleep(1)
        self.unrealcv.set_obj_rotation(self.injured_agent, self.injured_player_pose[3:])
        self.unrealcv.set_obj_location(self.injured_agent, self.injured_player_pose[:3])
        random_color = color = np.random.randint(100, 255, 3) #assign the injured agent a color for mask mode
        self.unrealcv.set_obj_color(self.injured_agent, random_color)

        time.sleep(1)
        self.unrealcv.set_obj_rotation(self.stretcher, self.rescue_pose[3:])
        self.unrealcv.set_obj_location(self.stretcher, self.rescue_pose[:3])
        time.sleep(1)
        self.unrealcv.set_phy(self.ambulance, 1)
        self.unrealcv.set_obj_rotation(self.ambulance, self.ambulance_pose[3:])
        self.unrealcv.set_obj_location(self.ambulance, self.ambulance_pose[:3])
        time.sleep(1)
        self.unrealcv.set_phy(self.ambulance, 0)
        # self.unrealcv.init_mask_color(self.injured_agent)
        # self.unrealcv.check_visibility(self.cam_list[self.protagonist_id], self.injured_agent)
        time.sleep(1)
        return states,info

    def reward(self, metrics):
        # individual reward
        if 'individual' in self.reward_type:
            if 'sparse' in self.reward_type:
                rewards = metrics['reach']  # only the agent who reach the target get the reward
            else:
                rewards = 1 - metrics['dis_each']/self.distance_threshold - np.fabs(metrics['ori_each'])/180 + metrics['reach']
        elif 'shared' in self.reward_type:
            if 'sparse' in self.reward_type:
                rewards = metrics['reach'].max()
            else:
                rewards = 1 - metrics['dis_min']/self.distance_threshold + metrics['reach'].max()
        else:
            raise ValueError('reward type is not defined')
        return rewards

    def rescue_metrics(self, objs_pose, target_loc):
        # compute the relative relation (distance, collision) among agents for rewards and evaluation metrics
        info = dict()
        relative_pose = []
        for obj_pos in objs_pose:
            obs, distance, direction = self.get_relative(obj_pos, target_loc)
            relative_pose.append(np.array([distance, direction]))
        relative_pose = np.array(relative_pose)
        relative_dis = relative_pose[:, 0]
        relative_ori = relative_pose[:, 1]
        reach_mat = np.zeros_like(relative_dis)
        reach_mat[np.where(relative_dis < self.distance_threshold)] = 1
        reach_mat[np.where(np.fabs(relative_ori) > 45)] = 0  # collision should be at the front view

        info['reach'] = reach_mat
        info['dis_min'] = relative_dis.min(-1)  # minimal distance from agents to target
        info['dis_each'] = relative_dis  # distance from agents to target
        info['ori_each'] = relative_ori  # orientation from agents to target
        info['picked']  = self.unrealcv.Is_picked(self.injured_agent)
        return info
