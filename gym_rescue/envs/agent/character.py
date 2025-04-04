from unrealcv.api import UnrealCv_API
import numpy as np
import math
import time
import json
import re
from io import BytesIO
import PIL.Image
class Character_API(UnrealCv_API):
    def __init__(self, port=9000, ip='127.0.0.1', resolution=(160, 120), comm_mode='tcp'):
        super(Character_API, self).__init__(port=port, ip=ip, resolution=resolution, mode=comm_mode)
        self.obstacles = []
        self.targets = []
        self.img_color = np.zeros((resolution[1], resolution[0], 3))
        self.img_depth = np.zeros((resolution[1], resolution[0], 1))
        self.animation_dict = {
            'stand': self.set_standup,
            'jump': self.set_jump,
            'crouch': self.set_crouch,
            'liedown': self.set_liedown,
            'open_door': self.set_open_door,
            'enter_vehicle': self.enter_exit_car,
            'carry':self.carry_body,
            'drop':self.drop_body
        }

    def config_ue(self, resolution=(320, 240), quality=1, disable_all_screen_messages=True, Lumen=False):
        """
        Configure Unreal Engine settings.

        Args:
            resolution (tuple): The resolution of the display window.
            quality (bool): The quality of the display window. From 0 to 3.
            disable_all_screen_messages (bool): Whether to disable all screen messages. Default is True.
        """
        self.check_connection()
        [w, h] = resolution
        self.client.request(f'vrun setres {w}x{h}w', -1)  # set resolution of the display window
        if disable_all_screen_messages:
            self.client.request('DisableAllScreenMessages', -1)  # disable all screen messages
        if quality > 0:
            self.client.request(f'vrun sg.ShadowQuality {quality}', -1)  # set shadow quality to low
            self.client.request(f'vrun sg.TextureQuality {quality}', -1)  # set texture quality to low
            self.client.request(f'vrun sg.EffectsQuality {quality}', -1)  # set effects quality to low
        if Lumen:
            self.client.request('vset /camera/1/illumination Lumen', -1)
            self.client.request('vset /camera/1/reflection Lumen', -1)
        time.sleep(0.1)

    def init_mask_color(self, targets=None):
        if targets == 'all':
            self.targets = self.get_objects()
            self.color_dict = self.build_color_dict(self.targets)
        elif targets is not None:
            self.targets = targets
            self.color_dict = self.build_color_dict(self.targets)

    def get_observation(self, cam_id, observation_type, mode='bmp'):
        if observation_type == 'Color':
            self.img_color = state = self.get_image(cam_id, 'lit', mode)
        elif observation_type == 'Mask':
            self.img_color = state = self.get_image(cam_id, 'object_mask', mode)
        elif observation_type == 'Depth':
            self.img_depth = state = self.get_depth(cam_id)
        elif observation_type == 'Rgbd':
            state = self.get_image_multimodal(cam_id, ['lit', 'depth'], [mode, 'npy'])
            self.img_color = state[..., :3]
            self.img_depth = state[..., 3:]
            state = np.append(self.img_color, self.img_depth, axis=2)
        elif observation_type == 'Gray':
            self.img_color = self.read_image(cam_id, 'lit', mode)
            self.img_gray = self.img_color.mean(2)
            self.img_gray = np.expand_dims(self.img_gray, -1)
        elif observation_type == 'Pose':
            state = self.get_cam_pose(cam_id) # camera 6D pose
        return state

    # functions for character setting
    def set_max_speed(self, player, speed, return_cmd=False):
        # set the max velocity of the agent object
        cmd = f'vbp {player} set_speed {speed}'
        res = None
        while res is None:
            res = self.client.request(cmd)
        return speed

    def set_acceleration(self, player, acc):
        cmd = f'vbp {player} set_acc {acc}'
        res = None
        while res is None:
            res = self.client.request(cmd.format(player=player, acc=acc))
        return acc

    def set_appearance(self, player, id):
        # set the appearance of the agent object, id: 0~19
        cmd = f'vbp {player} set_app {id}'
        res = None
        while res is None:
            res = self.client.request(cmd.format(player=player, id=id), -1)
        return id

    def move_cam_2d(self, cam_id, angle, distance):
        # move the camera in 2D as a mobile agent
        self.move_cam_forward(cam_id, angle, distance, height=0, pitch=0)

    def get_speed(self, player):
        cmd = f'vbp {player} get_speed'
        res = None
        while res is None:
            res = self.client.request(cmd)
        return self.decoder.string2vector(res)[0]

    def get_angle(self, player):
        cmd = f'vbp {player} get_angle'
        res = None
        while res is None:
            res = self.client.request(cmd)
        return self.decoder.string2vector(res)[0]

    def reset_player(self, player):
        cmd = f'vbp {player} reset'
        res=None
        while res is None:
            res = self.client.request(cmd)

    def set_phy(self, obj, state):
        cmd = f'vbp {obj} set_phy {state}'
        res=None
        while res is None:
            res = self.client.request(cmd, -1)

    def simulate_physics(self, objects):
        res = [self.set_phy(obj, 1) for obj in objects]

    def set_move_bp(self, player, params, return_cmd=False):
        '''
        new move function, can adapt to different number of params
        2 params: [v_angle, v_linear], used for agents moving in plane, e.g. human, car, animal
        4 params: [v_ x, v_y, v_z, v_yaw], used for agents moving in 3D space, e.g. drone
        '''
        params_str = ' '.join([str(param) for param in params])
        cmd = f'vbp {player} set_move {params_str}'
        if return_cmd:
            return cmd
        res = None
        while res is None:
            res = self.client.request(cmd, -1)

    # functions for character actions
    def set_jump(self, player, return_cmd=False):
        cmd = f'vbp {player} set_jump'
        if return_cmd:
            return cmd
        res = None
        while res is None:
            res = self.client.request(cmd, -1)

    def set_crouch(self, player, return_cmd=False):
        cmd = f'vbp {player} set_crouch'
        if return_cmd:
            return cmd
        res = None
        while res is None:
            res = self.client.request(cmd, -1)

    def set_liedown(self, player, directions=(100, 100), return_cmd=False):
        frontback = directions[0]
        leftright = directions[1]
        cmd = f'vbp {player} set_liedown {frontback} {leftright}'
        if return_cmd:
            return cmd
        res = None
        while res is None:
            res = self.client.request(cmd, -1)

    def set_standup(self, player, return_cmd=False):
        cmd = f'vbp {player} set_standup'
        if return_cmd:
            return cmd
        res = None
        while res is None:
            res = self.client.request(cmd, -1)

    def set_animation(self, player, anim_id, return_cmd=False):
        return self.animation_dict[anim_id](player, return_cmd=return_cmd)

    def get_hit(self, player):
        cmd = f'vbp {player} get_hit'
        res = None
        while res is None:
            res = self.client.request(cmd)
        if '1' in res:
            return True
        if '0' in res:
            return False

    def set_random(self, player, value=1):
        cmd = f'vbp {player} set_random {value}'
        res=None
        while res is None:
            res = self.client.request(cmd, -1)

    def set_interval(self, player, interval):
        cmd = f'vbp {player} set_interval {interval}'
        res = None
        while res is None:
            res = self.client.request(cmd, -1)

    def init_objects(self, objects):
        self.objects_dict = dict()
        for obj in objects:
            # print (obj)
            self.objects_dict[obj] = self.get_obj_location(obj)
        return self.objects_dict

    def random_obstacles(self, objects, img_dirs, num, area, start_area, texture=False):
        sample_index = np.random.choice(len(objects), num, replace=False)
        for id in sample_index:
            obstacle = objects[id]
            self.obstacles.append(obstacle)
            # texture
            if texture:
                img_dir = img_dirs[np.random.randint(0, len(img_dirs))]
                self.set_texture(obstacle, (1, 1, 1), np.random.uniform(0, 1, 3), img_dir, np.random.randint(1, 4))
            # scale
            # self.set_obj_scale(obstacle, np.random.uniform(0.3, 3, 3))
            self.set_obj_scale(obstacle, np.random.uniform(0.5, 4, 3))

            # location
            obstacle_loc = [start_area[0], start_area[2], 0]
            while start_area[0] <= obstacle_loc[0] <= start_area[1] and start_area[2] <= obstacle_loc[1] <= start_area[3]:
                obstacle_loc[0] = np.random.uniform(area[0]+100, area[1]-100)
                obstacle_loc[1] = np.random.uniform(area[2]+100, area[3]-100)
                obstacle_loc[2] = np.random.uniform(area[4], area[5]) -100
            self.set_obj_location(obstacle, obstacle_loc)
            time.sleep(0.01)

    def clean_obstacles(self):
        for obj in self.obstacles:
            self.set_obj_location(obj, self.objects_dict[obj])
        self.obstacles = []

    def new_obj(self, obj_class_name, obj_name, loc, rot=[0, 0, 0]):
        # spawn, set obj pose, enable physics
        [x, y, z] = loc
        [pitch, yaw, roll] = rot
        if obj_class_name =="bp_character_C" or obj_class_name =="target_C":
            cmd = [f'vset /objects/spawn {obj_class_name} {obj_name}',
                   f'vset /object/{obj_name}/location {x} {y} {z}',
                   f'vset /object/{obj_name}/rotation {pitch} {yaw} {roll}',
                   f'vbp {obj_name} set_phy 0'
                   ]
        else:
            cmd = [f'vset /objects/spawn {obj_class_name} {obj_name}',
                   f'vset /object/{obj_name}/location {x} {y} {z}',
                   f'vset /object/{obj_name}/rotation {pitch} {yaw} {roll}',
                   f'vbp {obj_name} set_phy 1'
                   ]
        self.client.request(cmd, -1)
        return obj_name

    def set_cam(self, obj, loc=[0, 30, 70], rot=[0, 0, 0], return_cmd=False):
        # set the camera pose relative to a actor
        x, y, z = loc
        roll, pitch, yaw = rot
        cmd = f'vbp {obj} set_cam {x} {y} {z} {roll} {pitch} {yaw}'
        if return_cmd:
            return cmd
        res = self.client.request(cmd, -1)
        return res

    def adjust_fov(self, cam_id, delta_fov, min_max=[45, 135]):  # increase/decrease fov
        return self.set_fov(cam_id, np.clip(self.cam[cam_id]['fov']+delta_fov, min_max[0], min_max[1]))

    def stop_car(self, obj):
        cmd = f'vbp {obj} set_stop'
        res = self.client.request(cmd, -1)
        return res

    def nav_to_goal(self, obj, loc): # navigate the agent to a goal location
        # Assign the agent a navigation goal, and use Navmesh to automatically control its movement to reach the goal via the shortest path.
        # The goal should be reachable in the environment.
        x, y, z = loc
        cmd = f'vbp {obj} nav_to_goal {x} {y} {z}'
        res = self.client.request(cmd, -1)
        return res

    def nav_to_goal_bypath(self, obj, loc): # navigate the agent to a goal location
        # Assign the agent a navigation goal, and use Navmesh to automatically control its movement to reach the goal via the shortest path.
        # The goal should be reachable in the environment.
        x, y, z = loc
        cmd = f'vbp {obj} nav_to_goal {x} {y} {z}'
        res = self.client.request(cmd, -1)
        return res

    def nav_to_random(self, obj, radius, loop): # navigate the agent to a random location
        # Agent randomly selects a point within its own radius range for navigation.
        # The loop parameter controls whether continuous navigation is performed.（True for continuous navigation).
        # Return with the randomly sampled location.
        cmd = f'vbp {obj} nav_random {radius} {loop}'
        res = self.client.request(cmd)
        return res

    def nav_to_obj(self, obj, target_obj, distance=200): # navigate the agent to a target object
        # Assign the agent a navigation goal, and use Navmesh to automatically control its movement to reach the goal via the shortest path.
        cmd = f'vbp {obj} nav_to_target {target_obj} {distance}'
        res = self.client.request(cmd, -1)
        return res

    def nav_random(self, player, radius, loop): # navigate the agent to a random location
        # Agent randomly selects a point within its own radius range for navigation.
        # The loop parameter controls whether continuous navigation is performed.（True for continuous navigation).
        # Return with the randomly sampled location.
        cmd = f'vbp {player} nav_random {radius} {loop}'
        res = self.client.request(cmd)
        return self.decoder.string2vector(res)

    def generate_nav_goal(self, player, radius_max,radius_min):  # navigate the agent to a random location
        # Agent randomly selects a point within its own radius range for navigation.
        # The loop parameter controls whether continuous navigation is performed.（True for continuous navigation).
        # Return with the randomly sampled location.
        cmd = f'vbp {player} generate_nav_goal {radius_max} {radius_min} '
        res = self.client.request(cmd)
        answer_dict = json.loads(res)
        try:
            loc = answer_dict["nav_goal"]
        except:
            loc = answer_dict["Nav_goal"]
        coordinates = re.findall(r"[-+]?\d*\.\d+|\d+", loc)
        # Convert the numbers to floats and store them in an array
        coordinates = [float(coordinate) for coordinate in coordinates]
        return coordinates[0],coordinates[1],coordinates[2]

    def set_max_nav_speed(self, obj, max_vel): # set the maximum navigation speed of the car
        cmd = f'vbp {obj} set_nav_speed {max_vel}'
        res = self.client.request(cmd, -1)
        return res

    def enter_exit_car(self, obj, player_index):
        # enter or exit the car for a player.
        # If the player is already in the car, it will exit the car. Otherwise, it will enter the car.
        cmd = f'vbp {obj} enter_exit_car {player_index}'
        res = self.client.request(cmd, -1)
        return res

    def set_open_door(self, player, return_cmd=False):
        # state: 0 close, 1 open
        cmd = f'vbp {player} set_open_door 1'
        if return_cmd:
            return cmd
        else:
            self.client.request(cmd, -1)
    def carry_body(self,player,return_cmd=False):
        cmd = f'vbp {player} carry_body'
        if return_cmd:
            return cmd
        else:
            self.client.request(cmd, -1)

    def drop_body(self,player, return_cmd=False):
        cmd = f'vbp {player} drop_body'
        if return_cmd:
            return cmd
        else:
            self.client.request(cmd, -1)

    def Is_picked(self,player, return_cmd = False):
        # check if the player is picked up
        cmd = f'vbp {player} is_picked'
        if return_cmd:
            return cmd
        else:
            res = self.client.request(cmd)
            if '1' in res:
                return True
            if '0' in res:
                return False
    def is_carrying(self,player, return_cmd = False):
        # check if the player is carrying a body
        cmd = f'vbp {player} is_carrying'
        if return_cmd:
            return cmd
        else:
            res = self.client.request(cmd)
            if '1' in res:
                return True
            if '0' in res:
                return False

    def set_viewport(self, player):
        # set the game window to the player's view
        cmd = f'vbp {player} set_viewport'
        res = self.client.request(cmd, -1)
        return res

    def get_pose_img_batch(self, objs_list, cam_ids, img_flag=[False, True, False, False]):
        # get pose and image of objects in objs_list from cameras in cam_ids
        cmd_list = []
        decoder_list = []
        [use_cam_pose, use_color, use_mask, use_depth] = img_flag
        for obj in objs_list:
            cmd_list.extend([self.get_obj_location(obj, True),
                             self.get_obj_rotation(obj, True)])

        for cam_id in cam_ids:
            if cam_id < 0:
                continue
            if use_cam_pose:
                cmd_list.extend([self.get_cam_location(cam_id, return_cmd=True),
                                 self.get_cam_rotation(cam_id, return_cmd=True)])
            if use_color:
                cmd_list.append(self.get_image(cam_id, 'lit', 'bmp', return_cmd=True))
            if use_mask:
                cmd_list.append(self.get_image(cam_id, 'object_mask', 'bmp', return_cmd=True))
            if use_depth:
                cmd_list.append(f'vget /camera/{cam_id}/depth npy')
                # cmd_list.append(self.get_image(cam_id, 'depth', 'bmp', return_cmd=True))

        decoders = [self.decoder.decode_map[self.decoder.cmd2key(cmd)] for cmd in cmd_list]
        res_list = self.batch_cmd(cmd_list, decoders)
        obj_pose_list = []
        cam_pose_list = []
        img_list = []
        mask_list = []
        depth_list = []
        # start to read results
        start_point = 0
        for i, obj in enumerate(objs_list):
            obj_pose_list.append(res_list[start_point] + res_list[start_point+1])
            start_point += 2
        for i, cam_id in enumerate(cam_ids):
            # print(cam_id)
            if cam_id < 0:
                img_list.append(np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8))
                continue
            if use_cam_pose:
                cam_pose_list.append(res_list[start_point] + res_list[start_point+1])
                start_point += 2
            if use_color:
                # image = self.decoder.decode_bmp(res_list[start_point])
                img_list.append(res_list[start_point])
                start_point += 1
            if use_mask:
                # image = self.decoder.decode_bmp(res_list[start_point])
                mask_list.append(res_list[start_point])
                start_point += 1
            if use_depth:
                # image = 1 / self.decoder.decode_depth(res_list[start_point],bytesio=False)
                # image = self.decoder.decode_depth(res_list[start_point],bytesio=False)
                image = self.get_depth(cam_id,show=False)
                # image = np.expand_dims(image, axis=-1)
                depth_list.append(image)  # 500 is the default max depth of most depth cameras
                # depth_list.append(res_list[start_point])  # 500 is the default max depth of most depth cameras
                start_point += 1

        return obj_pose_list, cam_pose_list, img_list, mask_list, depth_list

    # Domain Randomization Functions: randomize texture
    def set_texture(self, player, color=(1, 1, 1), param=(0, 0, 0), picpath=None, tiling=1, e_num=0): #[r, g, b, meta, spec, rough, tiling, picpath]
        param = param / param.max()
        r, g, b = color
        meta, spec, rough = param
        cmd = f'vbp {player} set_mat {e_num} {r} {g} {b} {meta} {spec} {rough} {tiling} {picpath}'
        self.client.request(cmd, -1)

    def set_light(self, obj, direction, intensity, color): # param num out of range
        [roll, yaw, pitch] = direction
        color = color / color.max()
        [r, g, b] = color
        cmd = f'vbp {obj} set_light {roll} {yaw} {pitch} {intensity} {r} {g} {b}'
        self.client.request(cmd, -1)

    def random_texture(self, backgrounds, img_dirs, num=5):
        if num < 0:
            sample_index = range(len(backgrounds))
        else:
            sample_index = np.random.choice(len(backgrounds), num, replace=False)
        for id in sample_index:
            obj = backgrounds[id]
            img_dir = img_dirs[np.random.randint(0, len(img_dirs))]
            self.set_texture(obj, (1, 1, 1), np.random.uniform(0, 1, 3), img_dir, np.random.randint(1, 4))
            time.sleep(0.03)

    def random_player_texture(self, player, img_dirs, num):
        sample_index = np.random.choice(5, num)
        for id in sample_index:
            img_dir = img_dirs[np.random.randint(0, len(img_dirs))]
            self.set_texture(player, (1, 1, 1), np.random.uniform(0, 1, 3),
                             img_dir, np.random.randint(2, 6), id)
            time.sleep(0.03)

    def random_character(self, player):  # appearance, speed, acceleration
        self.set_max_speed(player, np.random.randint(40, 100))
        self.set_acceleration(player, np.random.randint(100, 300))

    def random_lit(self, light_list):
        for lit in light_list:
            if 'sky' in lit:
                self.set_skylight(lit, [1, 1, 1], np.random.uniform(1, 10))
            else:
                lit_direction = np.random.uniform(-1, 1, 3)
                if 'directional' in lit:
                    lit_direction[0] = lit_direction[0] * 60
                    lit_direction[1] = lit_direction[1] * 80
                    lit_direction[2] = lit_direction[2] * 60
                else:
                    lit_direction *= 180
                self.set_light(lit, lit_direction, np.random.uniform(1, 5), np.random.uniform(0.3, 1, 3))

    def set_skylight(self, obj, color, intensity): # param num out of range
        [r, g, b] = color
        cmd = f'vbp {obj} set_light {r} {g} {b} {intensity}'
        self.client.request(cmd, -1)

    def get_obj_speed(self,obj):
        cmd = f'vbp {obj} get_speed'
        res = self.client.request(cmd)
        answer_dict = json.loads(res)
        speed = float(answer_dict["Speed"])
        return speed

    def check_visibility(self, tracker_cam_id, target_obj):
        # get the percentage of the target object visible in the tracker camera
        mask = self.get_image(tracker_cam_id, 'object_mask')
        print(mask.shape)
        mask, bbox = self.get_bbox(mask, target_obj, normalize=False)
        mask_percent = mask.sum()/(mask.shape[0]*mask.shape[1])
        return mask_percent