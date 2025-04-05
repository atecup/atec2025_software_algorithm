from agent_3_20.prompt import *
from agent_3_20.api import *
import os
import re
import argparse
import gym
import cv2
import time
import numpy as np
from gym_rescue.envs.wrappers import time_dilation, early_done, monitor, augmentation, configUE
import base64
from PIL import Image
import io
from openai import OpenAI
from datetime import datetime
from collections import deque


from openai import OpenAI
import random
import itertools
import time
import json
import http.client
from datetime import datetime
import csv
from typing import List, Dict, Optional
import os
import requests




class agt:

    def __init__(self, env, clue):
        self.clue = clue
        self.landmark = []
        self.landmark_back = []
        self.target = ""
        self.action = ""

        self.env = env
        self.obs, self.info = self.env.reset()
        self.obs, self.rewards, self.termination, self.truncation, self.info = self.env.step([([0, 0], 0, 0)])
        self.phase = 0
        return

    def start(self):
        self.analyse_initial_clue()
        print(f"LANDMARK LIST: {', '.join(self.landmark)}")
        move_fail_count = 0
        t0 = time.time()
        time_count = 1

        while True:
            t1 = time.time()
            print('Time cost:',t1-t0)
            if self.phase == 0 or self.phase == 1:
                # search the injured person
                while True:
                    search = self.search_landmark()
                    if search == True: # landmark exists
                        step = 0
                        while step < 2:
                            if self.move_to_landmark() == False:
                                step += 2
                            step += 1
                    
                    if self.target == "injured person" or self.target == "Injured person":
                        self.phase = 2
                        print("[SPEAK] I have found the injured person! Now I will try to rescue them.")
                        for _ in range(2):
                            self.execute_action(0, 50, 0, 0)
                        break
                    
                    if search == False:
                        self.search_move()

            elif self.phase == 2:  
                # rescue the injured person
                while True:
                    if self.info['picked'] == 1:
                        self.phase = 3
                        print("[SPEAK] I have rescued them. It's time to go back.")
                        break
                    elif self.move_to_landmark() == False:
                        self.execute_action(0, 20, 0, 0)
                        self.phase = 1
                        self.target = ""
                        print("[SPEAK] I lost the injured person. Let me try to search again.")
                        break



            elif self.phase == 3:
                # search the stretcher
                self.landmark = self.landmark_back
                while True:
                    search = self.search_landmark()

                    if self.target == "stretcher" or self.target == "Stretcher":
                        self.phase = 4
                        print("[SPEAK] I found the stretcher. Let me place the injured person on it.")
                        self.execute_action(0, 50, 0, 0)
                        break

                    if search == True: # 视野内有landmark
                        step = 0
                        while step < 3:
                            step += 1
                            if step == 1 and self.move_obstacle() == True:
                                if self.move_obstacle() == True:
                                    break
                                continue
                            if self.move_to_landmark() == False:
                                move_fail_count += 1
                                if move_fail_count >= 3:
                                    self.search_move()
                                break
                            else:
                                move_fail_count = 0
                        
                    else:
                        self.search_move()


            elif self.phase == 4:
                # place the person on the stretcher
                while True:
                    if self.move_to_landmark() == False:
                        self.execute_action(0, -80, 0, 0)
                        self.execute_action(0, -80, 0, 0)
                        self.execute_action(0, 0, 0, 4)
                        self.phase = 1
                        print("[SPEAK] I lost the stretcher. Let me search the person and rescue again.")
                        break
            t1 = time.time()
            if t1 - t0 > 180:
                self.truncation = True
            if self.truncation:
                 print("failed. time costed: {t1 - t0}")
                 break

            if self.termination:
                print("successed. time costed: {t1 - t0}")
                break





# ===========================================================================================================
# ===========================================================================================================


#                                                  WORKFLOW


# ===========================================================================================================
# ===========================================================================================================


    def analyse_initial_clue(self):
        prompt = initial_clue_prompt(self.clue)
        max_retries=3
        for attempt in range(max_retries):
            try:
                res = call_api_llm(prompt=prompt)
                print(res)
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>\s*<c>(.*?)</c>', re.DOTALL)
                match = pattern.search(res)

                if match:
                    direction = match.group(2).strip()
                    if direction == "left":
                        for _ in range(3):
                            self.execute_action(-30, 0, 0, 0)
                    elif direction == "right":
                        for _ in range(3):
                            self.execute_action(30, 0, 0, 0)
                    elif direction == "back":
                        for _ in range(6):
                            self.execute_action(30, 0, 0, 0)
                    elif "front" in direction and "left" in direction:
                        self.execute_action(-30, 0, 0, 0)
                    elif "front" in direction and "right" in direction:
                        self.execute_action(30, 0, 0, 0)


                    self.landmark = match.group(3).strip().split(",")
                    self.landmark = ["injured person" if landmark.strip().lower() == "injured woman" or landmark.strip().lower() == "injured man"
                                else landmark.strip() for landmark in self.landmark]
                    self.landmark = [landmark for landmark in self.landmark if "tretcher" not in landmark and "mbulance" not in landmark]
                    # landmark_back = self.landmark[::-1] + ['ambulance','stretcher']
                    # self.landmark_back = landmark_back[1:]
                    self.landmark_back = ['ambulance','stretcher']
                    return

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[CLUE ANALYSE] Failed after {max_retries} attempts: {str(e)}")




    def search_landmark(self):
        self.action = "search_landmark"
        if self.phase == 0:
            self.phase = 1
            prompt = search_prompt_begin(self.landmark)
            max_retries=3
            for attempt in range(max_retries):
                try:
                    res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.obs))
                    print(f"[SEARCH] \n {res} \n\n\n")
                    pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL)
                    match = pattern.search(res)

                    if match:
                        analysis = match.group(1).strip()
                        landmark = match.group(2).strip()

                        if landmark != "None" and landmark != "NONE":
                            self.target = landmark
                            self.execute_action(0, 50, 0, 1)
                            return True
                        else:
                            return False

                except Exception as e:
                    if attempt == max_retries - 1:
                        raise ValueError(f"[SEARCH] Failed after {max_retries} attempts: {str(e)}")


        elif self.phase == 3 or self.phase == 4:
            prompt = search_prompt_back(self.landmark)
            around_image = self.observe()
            max_retries=3
            for attempt in range(max_retries):
                try:
                    res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(around_image))
                    print(f"[SEARCH] \n {res} \n\n\n")
                    pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>\s*<c>(.*?)</c>', re.DOTALL)
                    match = pattern.search(res)

                    if match:
                        analysis = match.group(1).strip()
                        landmark = match.group(2).strip()
                        side = match.group(3).strip()

                        if landmark != "None" and landmark != "NONE":
                            self.target = landmark
                            if side == "left":
                                for _ in range(3):
                                    self.execute_action(-30, 0, 0, 0)
                            elif side == "right":
                                for _ in range(3):
                                    self.execute_action(30, 0, 0, 0)
                            elif side == "back":
                                for _ in range(6):
                                    self.execute_action(30, 0, 0, 0)
                            self.execute_action(0, 50, 0, 0)
                            return True
                        else:
                            return False

                except Exception as e:
                    if attempt == max_retries - 1:
                        raise ValueError(f"[SEARCH] Failed after {max_retries} attempts: {str(e)}")

        else:
            prompt = search_prompt(self.landmark)
            around_image = self.observe()
            max_retries=3
            for attempt in range(max_retries):
                try:
                    res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(around_image))
                    print(f"[SEARCH] \n {res} \n\n\n")
                    pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>\s*<c>(.*?)</c>', re.DOTALL)
                    match = pattern.search(res)

                    if match:
                        analysis = match.group(1).strip()
                        landmark = match.group(2).strip()
                        side = match.group(3).strip()

                        if landmark != "None" and landmark != "NONE":
                            self.target = landmark
                            if side == "left":
                                for _ in range(3):
                                    self.execute_action(-30, 0, 0, 0)
                            elif side == "right":
                                for _ in range(3):
                                    self.execute_action(30, 0, 0, 0)
                            self.execute_action(0, 50, 0, 0)
                            return True
                        else:
                            return False

                except Exception as e:
                    if attempt == max_retries - 1:
                        raise ValueError(f"[SEARCH] Failed after {max_retries} attempts: {str(e)}")

    



    def move_to_landmark(self):
        self.action = "move_to_landmark"
        prompt = move_forward_prompt(self.target)
        max_retries=3
        for attempt in range(max_retries):
            try:
                res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.add_vertical_lines(self.obs)))
                print(f"[MOVING] \n {res} \n\n\n")
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL)
                match = pattern.search(res)

                if match:
                    tag = match.group(1).strip()
                    direction = match.group(2).strip()

                    if tag == "no" or tag == "No" or tag == "NO":
                        return False
                    elif tag == "yes" or tag == "Yes" or tag == "YES":
                        if direction == "left":
                            self.execute_action(-30, 0, 0, 0)
                        elif direction == "right":
                            self.execute_action(30, 0, 0, 0)
                        elif direction == "middle":
                            self.execute_action(0, 100, 0, 0)
                        # if self.target != "injured person" and self.target != "Injured person" and self.target != "Stretcher" and self.target != "stretcher":
                        #     if self.phase == 1 or self.phase == 3:
                        #         for _ in range(2):
                        #             self.execute_action(0, 100, 0, 0)
                        self.execute_action(0, 100, 0, 0)
                        if self.info['picked'] != 1:
                            self.execute_action(0, 40, 0, 1)
                        return True

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[MOVE] Failed after {max_retries} attempts: {str(e)}")



    def move_obstacle(self):
        self.action = "move_obstacle"
        prompt = move_obstacle_prompt()
        self.execute_action(0, -50, 0, 0)
        self.execute_action(0, 0, 2, 0)
        
        max_retries=3
        for attempt in range(max_retries):
            try:
                res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.obs))
                print(f"[CHECKING OBSTACLE] \n {res} \n\n\n")
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL) # a：是否有障碍物 b：是什么障碍物
                match = pattern.search(res)

                if match:
                    tag = match.group(1).strip()
                    obstacle = match.group(2).strip()

                    if tag == "1":
                        # 随机向左或向右绕过障碍物
                        turn = random.choice([-30, 30])
                        self.execute_action(turn, 0, 0, 0)
                        self.execute_action(turn, 0, 0, 0)
                        for _ in range(5):
                            self.execute_action(0, 100, 0, 0)
                        self.execute_action(-1 * turn, 0, 0, 0)
                        self.execute_action(-1 * turn, 0, 0, 0)

                        return True

                    else:
                        return False
                

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[MOVE_OBSTACLE] Failed after {max_retries} attempts: {str(e)}")






    def search_move(self):
        self.action = "search_move"
        prompt = search_move_prompt()
        around_image = self.observe()

        max_retries=3
        for attempt in range(max_retries):
            try:
                res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(around_image))
                print(f"[SEARCH-MOVE] \n {res} \n\n\n")
                pattern = re.compile(r'<think>(.*?)</think>\s*<output>(.*?)</output>', re.DOTALL)
                match = pattern.search(res)

                if match:
                    output = match.group(2).strip()

                    if output == "left":
                        for _ in range(3):
                            self.execute_action(-30, 0, 0, 0)
                    elif output == "right":
                        for _ in range(3):
                            self.execute_action(30, 0, 0, 0)

                    for _ in range(3):
                        self.execute_action(0, 100, 0, 0)
                    if self.info["picked"] != 1:
                        self.execute_action(0, 50, 0, 1)

                    return

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[SEARCH_MOVE] Failed after {max_retries} attempts: {str(e)}")
                


    def random_move(self):
        self.action = "random_move"
        direction = random.choice([-1,1])
        if direction == -1:
            for _ in range(3):
                self.execute_action(-30, 0, 0, 0)
            self.execute_action(0, 50, 0, 0)
            for _ in range(3):
                self.execute_action(30, 0, 0, 0)

        elif direction == 1:
            for _ in range(3):
                self.execute_action(30, 0, 0, 0)
            self.execute_action(0, 50, 0, 0)
            for _ in range(3):
                self.execute_action(-30, 0, 0, 0)



    def observe(self):
        self.action = "observe"
        if self.phase == 3 or self.phase == 4:
            image_buffer = []
            for _ in range(4):
                image_buffer.append(self.obs.copy())
                for _ in range(3):
                    self.execute_action(30, 0, 0, 0)
            
            image = self.concatenate_images(image_buffer)
            return image
        else:
            image_buffer = []
            for _ in range(3):
                self.execute_action(-30, 0, 0, 0)
            for _ in range(2):
                image_buffer.append(self.obs.copy())
                for _ in range(3):
                    self.execute_action(30, 0, 0, 0)
            image_buffer.append(self.obs.copy())
            for _ in range(3):
                self.execute_action(-30, 0, 0, 0)
            
            image = self.concatenate_images(image_buffer)
            return image
    




# ===========================================================================================================
# ===========================================================================================================


#                                                    IMAGE


# ===========================================================================================================
# ===========================================================================================================



    def encode_image_array(self, image_array):
        # Convert the image array to a PIL Image object
        image = Image.fromarray(np.uint8(image_array))

        # Save the PIL Image object to a bytes buffer
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")  # You can change JPEG to PNG or other formats depending on your needs

        # Encode the bytes buffer to Base64
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_str



    def concatenate_images(self, image_list):


        height, width, channels = image_list[0].shape

        total_width = width * len(image_list)
        concatenated_image = np.zeros((height, total_width, channels), dtype=np.uint8)

        for i, img in enumerate(image_list):
            concatenated_image[:, i * width:(i + 1) * width, :] = img

        return concatenated_image
    



    def add_vertical_lines(self, image_array):
        h, w, c = image_array.shape

        line1 = w // 3
        line2 = w * 2 // 3
        line_color = (0, 0, 255)
        line_thickness = 2

        cv2.line(image_array, (line1, 0), (line1, h), line_color, line_thickness)
        cv2.line(image_array, (line2, 0), (line2, h), line_color, line_thickness)

        return image_array

# ===========================================================================================================
# ===========================================================================================================


#                                                  ACTION


# ===========================================================================================================
# ===========================================================================================================





    def execute_action(self, a, b, c, d):
        action = ([0, 0], 0, 0) 
        action = list(action) 
        action[0] = list(action[0])
        action[0][0] = a
        action[0][1] = b
        action[1] = c
        action[2] = d
        # action[0] = tuple(action[0])
        action = tuple(action) 
        # print(action)
        # import pdb
        # pdb.set_trace()
        self.obs, self.rewards, self.termination, self.truncation, self.info = self.env.step([action])
        time.sleep(1)
        # print("----------------------action down!-------------------------")
        cv2.imshow('mask', self.obs)
        cv2.waitKey(1)

        if self.target == 'injured person' or self.target == 'Injured person':
            if self.info['picked'] == 0:
                action = ([0, 0], 0, 0)  
                action = list(action)  
                action[0] = list(action[0])
                action[2] = 3
                action = tuple(action) 
                self.obs, self.rewards, self.termination, self.truncation, self.info = self.env.step([action])
                time.sleep(1)
                # print("----------------------action down!-------------------------")
                cv2.imshow('mask', self.obs)
                cv2.waitKey(1)


