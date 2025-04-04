import gym
from gym import Wrapper
import time
import cv2

class DisplayWrapper(Wrapper):
    def __init__(self, env, dynamic_top_down=True, fix_camera=False, get_bbox=False):
        super().__init__(env)
        self.dynamic_top_down = dynamic_top_down
        self.fix_camera = fix_camera
        self.get_bbox = get_bbox

    def step(self, action):
        obs, reward, termination, truncation, info = self.env.step(action) # take a step in the wrapped environment
        # set top_down camera
        env = self.env.unwrapped

        # for recording demo
        if self.get_bbox:
            mask = env.unrealcv.get_image(env.cam_list[env.protagonist_id], 'object_mask', 'bmp')
            mask, bbox = env.unrealcv.get_bbox(mask, env.unwrapped.injured_agent, normalize=False)
            self.show_bbox(env.img_show.copy(), bbox)
            info['bbox'] = bbox

        if self.dynamic_top_down:
            env.set_topview(info['Pose'][env.protagonist_id], env.cam_id[0]) # set top_down camera

        return obs, reward, termination, truncation, info # return the same results as the wrapped environment

    def reset(self, **kwargs):
        states = self.env.reset(**kwargs)
        env = self.env.unwrapped
        if self.fix_camera:
            # current_pose = env.unrealcv.get_cam_pose(env.cam_list[env.protagonist_id])
            center_pos = [self.ambulance_pose[0], self.ambulance_pose[1], self.ambulance_pose[2]+2000]
            env.set_topview(center_pos, env.cam_id[0])
        if self.get_bbox:
            self.bbox_init = []
            mask = env.unrealcv.get_image(env.cam_list[env.protagonist_id], 'object_mask', 'bmp')
            mask, bbox = env.unrealcv.get_bbox(mask, env.unwrapped.injured_agent, normalize=False)
            self.mask_percent = mask.sum()/(255 * env.resolution[0] * env.resolution[1])
            self.bbox_init.append(bbox)
        # cv2.imshow('init', env.img_show)
        # cv2.waitKey(1)
        return states # return the same results as the wrapped environment

    def show_bbox(self, img2disp, bbox):
        # im_disp = states[0][:, :, :3].copy()
        cv2.rectangle(img2disp, (int(bbox[0]), int(bbox[1])), (int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])), (0, 255, 0), 5)
        cv2.imshow('track_res', img2disp)
        cv2.waitKey(1)