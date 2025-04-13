from prompt_yoloVLM import *
from api_yoloVLM import *
import os
import re
#import argparse
#import gym
import cv2
import time
import numpy as np
import base64
from PIL import Image
import io
from datetime import datetime
from collections import deque
import random
from ultralytics import YOLO


class agent:

    def __init__(self,reference_text,reference_image):
        self.clue = ""
        self.landmark = []
        self.landmark_back = []
        self.target = ""
        self.action = ""
        
        # Initialize obs and info
        self.obs = None
        self.info = None

        self.phase = 0
        self.initialized = False
        
        # Create buffers for actions and observations
        self.clue = reference_text

        self.image_clue = reference_image

        self.person_text = ""
        self.landmark = []
        self.landmark_back = []
        self.target = ""
        self.action = ""

        # Initialize obs and info
        self.obs = None
        self.info = None

        self.phase = 0

        # Create buffers for actions and observations
        self.action_buffer = []
        self.obs_buffer = []
        self.image_buffer = []

        # State tracking
        self.current_step = 0
        self.move_fail_count = 0
        self.final_move_count = 0
        self.search_in_progress = False
        self.move_to_landmark_in_progress = False
        self.observe_in_progress = False
        self.move_obstacle_in_progress = False
        self.search_move_in_progress = False
        self.pending_action = None
        self.check = 0
        self.obs_storage_indices = []
        self.around_image = None

        # Add step counters for limiting move_to_landmark iterations
        self.move_steps = 0
        self.max_move_steps = 2  # Match original code limit of 2 iterations

        self.myinfo = {'picked': 0}
        
        return

    def predict(self, obs, info):
        # Add a 1-second delay at the beginning of predict
        time.sleep(1)

        # Store the current observation and info
        self.obs = obs
        self.info = info

        # Check if there are pending actions in the buffer
        if self.action_buffer:
            action = self.action_buffer.pop(0)

            # If this action should store an observation, mark it
            if self.obs_buffer and len(self.obs_buffer) > 0:
                if self.obs_buffer[0]:
                    # Store the current observation for future use
                    self.image_buffer.append(obs.copy())
                self.obs_buffer.pop(0)

            return action

        # Start the main logic chain if no pending actions
        if self.phase == 0:
            # Initialize by analyzing the clue
            return self._handle_initial_phase()

        elif self.phase == 1:
            # Search for the injured person
            return self._handle_search_phase()

        elif self.phase == 2:
            # Rescue the injured person
            return self._handle_rescue_phase()

        elif self.phase == 3:
            # Search for the stretcher/ambulance
            return self._handle_return_phase()

        elif self.phase == 4:
            # Place the person on the stretcher
            return self._handle_placement_phase()


        # Default action if nothing else to do
        return ([0, 0], 0, 0)

    def _handle_initial_phase(self):
        # First time through, analyze the clue
        if not self.landmark:
            self.analyse_initial_clue()
            self.analyse_initial_image()
            print(f"\n\n\nLANDMARK LIST: {', '.join(self.landmark)}")
            if self.action_buffer:
                return self.action_buffer.pop(0)
            # Directly search in the initial direction without rotating
        prompt = search_prompt_begin(self.landmark, self.person_text)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # res1 = call_api_vlm(prompt = """Please observe the image to determine if there is a relatively open area in front of you or if there are objects close enough for clear observation? Please only return "1" to indicate emptiness, or "0" to indicate not emptiness""", base64_image=self.encode_image_array(self.obs[0]))
                # if res1 == "1":
                #     for i in range(5):
                #         self.action_buffer.append(([0, 100], 0, 0))
                res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.obs))
                print(f"[SEARCH] \n {res} \n\n\n")
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL)
                match = pattern.search(res)

                if match:
                    analysis = match.group(1).strip()
                    landmark = match.group(2).strip()

                    if landmark != "None" and landmark != "NONE":
                        self.target = landmark
                        self.phase = 1
                        self.move_to_landmark_in_progress = True
                        # Queue up the initial forward movement
                        self.action_buffer.append(([0, 50], 0, 1))
                        # if "njured" not in self.target:
                        #     self.action_buffer.append(([0, 100], 0, 0))
                        #     self.action_buffer.append(([0, 100], 0, 0))
                        return ([0, 0], 0, 0)
                    else:
                        self.phase = 1
                        self.search_in_progress = True
                        return self._start_search()

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[SEARCH] Failed after {max_retries} attempts: {str(e)}")

            # If we get here, no landmark was found
            self.phase = 1
            self.search_in_progress = True
            return self._start_search()

        # Return default action if already initialized
        return ([0, 0], 0, 0)

    def _handle_search_phase(self):
        if self.info['picked'] == 1:
            self.phase = 3
            self.action_buffer.clear()
            self.obs_buffer.clear()
            print("[SPEAK] I have rescued them. It's time to go back.")
            return self._start_search()
        # If search is currently in progress, continue it
        if self.search_in_progress:
            result = self._process_search_result()
            self.search_in_progress = False

            if result == True:  # landmark exists
                self.move_to_landmark_in_progress = True
                self.move_steps = 0  # Reset step counter when starting new movement
                return self._start_move_to_landmark()
            else:
                # self.search_move_in_progress = True
                # return self._start_search_move()
                self.action_buffer.append(([0, 100], 0, 0))
                self.action_buffer.append(([0, 100], 0, 0))
                self.action_buffer.append(([0, 100], 0, 0))
                return self._start_search()

        # If moving to landmark is in progress
        elif self.move_to_landmark_in_progress:
            if self.move_steps > self.max_move_steps:
                # If reached max steps, go back to search
                self.move_to_landmark_in_progress = False
                self.search_in_progress = True
                return self._start_search()

            result = self._process_move_result()

            if result == False:
                self.move_fail_count += 1
                self.move_to_landmark_in_progress = False
                self.search_in_progress = True
                return self._start_search()

            # Check if we've reached max steps for move_to_landmark
            self.move_steps += 1

            if "injured" in self.target:
                self.phase = 2
                self.move_to_landmark_in_progress = False
                print("[SPEAK] I have found the injured person! Now I will try to rescue them.")

                # Queue up two forward actions
                self.action_buffer.append(([0, 50], 0, 3))
                self.obs_buffer.append(False)
                return ([0, 50], 0, 0)

            # Continue moving to landmark
            return self._start_move_to_landmark()

        # If neither search nor move is in progress, start search
        else:
            self.search_in_progress = True
            return self._start_search()

    def _handle_rescue_phase(self):
        # If person is picked, move to next phase
        if self.info['picked'] == 1:
            self.phase = 3
            print("[SPEAK] I have rescued them. It's time to go back.")
            return ([0, 0], 0, 0)

        # If moving to landmark is in progress
        elif self.move_to_landmark_in_progress:
            result = self._process_move_result()

            if result == False:
                self.move_to_landmark_in_progress = False
                self.phase = 1
                self.target = ""
                print("[SPEAK] I lost the injured person. Let me try to search again.")
                return ([0, 20], 0, 0)

            # Continue moving to landmark
            return self._start_move_to_landmark()

        # Start moving to landmark
        else:
            self.move_to_landmark_in_progress = True
            self.move_steps = 0  # Reset step counter
            return self._start_move_to_landmark()

    def _handle_return_phase(self):
        # Update landmarks for the return journey
        if not self.landmark_back:
            self.landmark = ['ambulance', 'orange bench']
            self.landmark_back = self.landmark

        # If search is in progress
        if self.search_in_progress:
            result = self._process_search_result()
            self.search_in_progress = False

            if "range" in self.target or "ench" in self.target:
                self.phase = 4
                print("[SPEAK] I found the stretcher. Let me place the injured person on it.")
                return self._start_move_to_landmark()

            if result == True:  # landmark exists in view
                if self.move_obstacle_in_progress:
                    obstacle_result = self._process_obstacle_result()
                    if obstacle_result == True:
                        self.move_obstacle_in_progress = False
                        return self._start_search()

                self.move_to_landmark_in_progress = True
                return self._start_move_to_landmark()
            else:
                self.search_in_progress = True
                return self._start_search()

        elif self.move_obstacle_in_progress:
            result = self._process_obstacle_result()
            self.move_obstacle_in_progress = False
            self.move_to_landmark_in_progress = True
            return self._start_move_to_landmark()

        # If moving to landmark is in progress
        elif self.move_to_landmark_in_progress:
            if self.move_steps > self.max_move_steps:
                # If reached max steps, go back to search
                self.move_to_landmark_in_progress = False
                self.search_in_progress = True
                return self._start_search()

            result = self._process_move_result()

            if result == False:
                self.move_to_landmark_in_progress = False
                self.search_in_progress = True
                return self._start_search()
            else:
                self.move_steps += 1

            # Keep trying to move to landmark
            return self._start_move_to_landmark()

        # Start the search process
        else:
            self.search_in_progress = True
            return self._start_search()

    # def _handle_placement_phase(self):
    #     # If moving to landmark is in progress
    #     if self.move_to_landmark_in_progress:
    #         result = self._process_move_result()
    #         self.final_move_count += 1

    #         if result == False or self.final_move_count >= 10:
    #             self.move_to_landmark_in_progress = False
    #             print("[SPEAK] I am going to place the injured person.")
    #             self.phase = 1
    #             # Queue backward movements and drop action
    #             self.action_buffer.append(([0, -100], 0, 0))
    #             self.action_buffer.append(([0, 0], 0, 4))
    #             return ([0, -100], 0, 0)

    #         # Continue moving to landmark
    #         return self._start_move_to_landmark()

    #     # Start moving to landmark
    #     else:
    #         self.move_to_landmark_in_progress = True
    #         return self._start_move_to_landmark()

    def _handle_placement_phase(self):
        # if moving to landmark is in progress
        if self.move_to_landmark_in_progress:
            result = self._process_move_result()
            self.final_move_count += 1

            if result == False:
                if self.check == 0:
                    self.check = 1
                    # move backward
                    self.action_buffer.clear()
                    self.obs_buffer.clear()
                    self.action_buffer.append(([0, -100], 0, 0))
                    self.obs_buffer.append(False)

                    # observe
                    self.action_buffer.append(([0, 0], 2, 0))
                    self.obs_buffer.append(True)  # observe to check

                    print("[SPEAK] let me check the stretcher")
                    return ([0, -100], 0, 0)
                elif self.check == 1:
                    self.check = 0
                    self.move_to_landmark_in_progress = False
                    self.search_in_progress = True
                    self.phase = 3
                    return self._start_search()
                elif self.check == 2:
                    self.action_buffer.append(([0, -100], 0, 0))
                    self.action_buffer.append(([0, -100], 0, 0))
                    self.action_buffer.append(([0, -50], 0, 4))
            else:
                if self.final_move_count >= 10:
                    self.move_to_landmark_in_progress = False
                    print("[SPEAK] I have moved too many times. Maybe I am stuck.")
                    self.phase = 3
                    self.action_buffer.append(([0, -50], 0, 0))
                    self.action_buffer.append(([0, 0], 0, 4))
                    return ([0, -100], 0, 0)
                if self.check == 1:
                    self.check = 2
            # keep moving
            return self._start_move_to_landmark()

        # move
        else:
            self.move_to_landmark_in_progress = True
            return self._start_move_to_landmark()

    # Helper methods to start various processes
    def _start_search(self):
        # Prepare for search by observing surroundings
        self.observe_in_progress = True
        self.around_image = None
        self.image_buffer = []

        # For phase 3 or 4, start rotation to observe surroundings
        self.action_buffer = []
        self.obs_buffer = []

        if self.phase == 3 or self.phase == 4:
            # Prepare for 4 observations while rotating
            for i in range(4):
                # For first observation, don't rotate
                if i == 0:
                    self.action_buffer.append(([0, 0], 0, 0))
                    self.obs_buffer.append(True)
                else:
                    # Add 3 rotations (first 2 don't save observations)
                    for j in range(3):
                        self.action_buffer.append(([30, 0], 0, 0))
                        # Only save observation on the last rotation of each set
                        self.obs_buffer.append(j == 2)
            for i in range(3):
                self.action_buffer.append(([30, 0], 0, 0))
                self.obs_buffer.append(False)

            # Return first action
            action = self.action_buffer.pop(0)
            should_save = self.obs_buffer.pop(0)
            if should_save:
                self.image_buffer.append(self.obs.copy())
            return action
        else:
            # First rotate left 3 times (don't save observations)
            for _ in range(3):
                self.action_buffer.append(([-30, 0], 0, 0))
                self.obs_buffer.append(False)

            # Now we're facing left - save this observation
            self.action_buffer.append(([0, 0], 0, 0))
            self.obs_buffer.append(True)

            # Rotate right 3 times to face forward (don't save)
            for _ in range(3):
                self.action_buffer.append(([30, 0], 0, 0))
                self.obs_buffer.append(False)

            # Now we're facing forward - save this observation
            self.action_buffer.append(([0, 0], 0, 0))
            self.obs_buffer.append(True)

            # Rotate right 3 more times to face right (don't save)
            for _ in range(3):
                self.action_buffer.append(([30, 0], 0, 0))
                self.obs_buffer.append(False)

            # Now we're facing right - save this observation
            self.action_buffer.append(([0, 0], 0, 0))
            self.obs_buffer.append(True)

            # Rotate left 3 times to return to center (don't save)
            for _ in range(3):
                self.action_buffer.append(([-30, 0], 0, 0))
                self.obs_buffer.append(False)

            # Get first action and observation status
            action = self.action_buffer.pop(0)
            should_save = self.obs_buffer.pop(0)
            if should_save:
                self.image_buffer.append(self.obs.copy())

            return action

    def _start_move_to_landmark(self):
        # This action will be handled immediately, the result processed on next prediction
        return ([0, 0], 0, 0)

    def _start_move_obstacle(self):
        # Prepare for obstacle check
        self.action_buffer = [
            ([0, -50], 0, 0),
            ([0, 0], 2, 0)
        ]
        self.obs_buffer = [False, False]

        # Return first action
        action = self.action_buffer.pop(0)
        self.obs_buffer.pop(0)
        return action

    def _start_search_move(self):
        # Prepare for search move
        self.observe_in_progress = True
        self.image_buffer = []

        # Set up observation sequence
        self.action_buffer = []
        self.obs_buffer = []

        if self.phase == 3 or self.phase == 4:
            # Similar to search but with a focus on movement
            for i in range(4):
                if i == 0:
                    self.action_buffer.append(([0, 0], 0, 0))
                    self.obs_buffer.append(True)
                else:
                    for j in range(3):
                        self.action_buffer.append(([30, 0], 0, 0))
                        self.obs_buffer.append(j == 2)

            # Get first action and observation status
            action = self.action_buffer.pop(0)
            should_save = self.obs_buffer.pop(0)
            if should_save:
                self.image_buffer.append(self.obs.copy())

            return action
        else:
            # First rotate left 3 times (don't save observations)
            for _ in range(3):
                self.action_buffer.append(([-30, 0], 0, 0))
                self.obs_buffer.append(False)

            # Now we're facing left - save this observation
            self.action_buffer.append(([0, 0], 0, 0))
            self.obs_buffer.append(True)

            # Rotate right 3 times to face forward (don't save)
            for _ in range(3):
                self.action_buffer.append(([30, 0], 0, 0))
                self.obs_buffer.append(False)

            # Now we're facing forward - save this observation
            self.action_buffer.append(([0, 0], 0, 0))
            self.obs_buffer.append(True)

            # Rotate right 3 more times to face right (don't save)
            for _ in range(3):
                self.action_buffer.append(([30, 0], 0, 0))
                self.obs_buffer.append(False)

            # Now we're facing right - save this observation
            self.action_buffer.append(([0, 0], 0, 0))
            self.obs_buffer.append(True)

            # Rotate left 3 times to return to center (don't save)
            for _ in range(3):
                self.action_buffer.append(([-30, 0], 0, 0))
                self.obs_buffer.append(False)

            # Get first action and observation status
            action = self.action_buffer.pop(0)
            should_save = self.obs_buffer.pop(0)
            if should_save:
                self.image_buffer.append(self.obs.copy())

            return action

    # Methods to process results of various actions
    def _process_search_result(self):
        # If we've collected observations, process them
        if self.image_buffer:
            self.around_image = self.concatenate_images(self.image_buffer)
            self.image_buffer = []

            if self.phase == 0:
                self.phase = 1
                prompt = search_prompt_begin(self.landmark, self.person_text)
                max_retries = 3
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
                                # Queue search confirmation action
                                self.action_buffer.append(([0, 100], 0, 1))
                                self.action_buffer.append(([0, 100], 0, 1))
                                self.obs_buffer.append(False)  # Don't save observation after this action
                                return True
                            else:
                                return False

                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise ValueError(f"[SEARCH] Failed after {max_retries} attempts: {str(e)}")

            elif self.phase == 3 or self.phase == 4:
                landmark = random.choice([self.landmark, ["orange bench"]])
                prompt = search_prompt_back(landmark, self.person_text)
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.around_image))
                        print(f"[SEARCH] \n {res} \n\n\n")
                        pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>\s*<c>(.*?)</c>', re.DOTALL)
                        match = pattern.search(res)

                        if match:
                            analysis = match.group(1).strip()
                            landmark = match.group(2).strip()
                            side = match.group(3).strip()

                            if landmark != "None" and landmark != "NONE":
                                self.target = landmark
                                # Queue appropriate rotation actions based on side
                                if side == "left":
                                    for _ in range(3):
                                        self.action_buffer.append(([-30, 0], 0, 0))
                                        self.obs_buffer.append(False)  # Don't save observations during rotation
                                elif side == "right":
                                    for _ in range(3):
                                        self.action_buffer.append(([30, 0], 0, 0))
                                        self.obs_buffer.append(False)  # Don't save observations during rotation
                                elif side == "back":
                                    for _ in range(6):
                                        self.action_buffer.append(([30, 0], 0, 0))
                                        self.obs_buffer.append(False)  # Don't save observations during rotation
                                # Add forward action
                                self.action_buffer.append(([0, 100], 0, 0))
                                self.action_buffer.append(([0, 100], 0, 0))
                                self.obs_buffer.append(False)  # Don't save observation after this action
                                return True
                            else:
                                return False

                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise ValueError(f"[SEARCH] Failed after {max_retries} attempts: {str(e)}")

            else:
                prompt = search_prompt(self.landmark, self.person_text)
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.around_image))
                        print(f"[SEARCH] \n {res} \n\n\n")
                        pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>\s*<c>(.*?)</c>', re.DOTALL)
                        match = pattern.search(res)

                        if match:
                            analysis = match.group(1).strip()
                            landmark = match.group(2).strip()
                            side = match.group(3).strip()

                            if landmark != "None" and landmark != "NONE":
                                self.target = landmark
                                self.action_buffer.append(([0, 10], 0, 3))
                                # Queue appropriate rotation actions based on side
                                if side == "left":
                                    for _ in range(3):
                                        self.action_buffer.append(([-30, 0], 0, 0))
                                        self.obs_buffer.append(False)  # Don't save observations during rotation
                                elif side == "right":
                                    for _ in range(3):
                                        self.action_buffer.append(([30, 0], 0, 0))
                                        self.obs_buffer.append(False)  # Don't save observations during rotation
                                # Add forward action

                                self.action_buffer.append(([0, 100], 0, 0))
                                self.obs_buffer.append(False)  # Don't save observation after this action
                                return True
                            else:
                                return False

                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise ValueError(f"[SEARCH] Failed after {max_retries} attempts: {str(e)}")

        return False

    def _process_move_result(self):
        prompt = move_forward_prompt(self.target, self.person_text)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = call_api_vlm(prompt=prompt,
                                   base64_image=self.encode_image_array(self.add_vertical_lines(self.obs)))
                print(f"[MOVING] \n {res} \n\n\n")
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL)
                match = pattern.search(res)

                if match:
                    tag = match.group(1).strip()
                    direction = match.group(2).strip()

                    if tag == "no" or tag == "No" or tag == "NO":
                        return False
                    elif tag == "yes" or tag == "Yes" or tag == "YES":
                        # Queue appropriate actions based on direction
                        if direction == "left":
                            self.action_buffer.append(([-20, 0], 0, 0))
                            self.action_buffer.append(([-20, 0], 0, 0))
                        elif direction == "right":
                            self.action_buffer.append(([20, 0], 0, 0))
                            self.action_buffer.append(([20, 0], 0, 0))
                        elif direction == "middle":
                            self.action_buffer.append(([0, 100], 0, 0))
                            self.action_buffer.append(([0, 100], 0, 0))
                        # forward action
                        self.action_buffer.append(([0, 50], 0, 3))
                        self.obs_buffer.append(False)  # Don't save observation during forward movement

                        # Special handling for injured person target
                        # if "njured" in self.target and self.info['picked'] == 0:
                        #     # Add pickup action (3) as in original code
                        #     self.action_buffer.append(([0, 0], 0, 3))
                        #     self.obs_buffer.append(False)  # Don't save observation during pickup

                        # Special handling for cross the door
                        if "door" in self.target:
                            self.action_buffer.append(([0, 0], 0, 5))
                            self.action_buffer.append(([0, 100], 0, 0))
                        # jump can fix many bugs!
                        if self.info['picked'] != 1:
                            self.action_buffer.append(([0, 40], 0, 1))
                            self.obs_buffer.append(False)  # Don't save observation during pickup check

                        return True

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[MOVE] Failed after {max_retries} attempts: {str(e)}")

        return False

    def _process_obstacle_result(self):
        prompt = move_obstacle_prompt()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.obs))
                print(f"[CHECKING OBSTACLE] \n {res} \n\n\n")
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL)
                match = pattern.search(res)

                if match:
                    tag = match.group(1).strip()
                    obstacle = match.group(2).strip()

                    if tag == "1":
                        # Queue random turning and movement to avoid obstacle
                        turn = random.choice([-30, 30])
                        self.action_buffer.append(([turn, 0], 0, 0))
                        self.obs_buffer.append(False)  # Don't save observation during turning

                        self.action_buffer.append(([turn, 0], 0, 0))
                        self.obs_buffer.append(False)  # Don't save observation during turning

                        # Add forward movements
                        for _ in range(5):
                            self.action_buffer.append(([0, 100], 0, 0))
                            self.obs_buffer.append(False)  # Don't save observations during forward movement

                        # Add return rotation
                        self.action_buffer.append(([-1 * turn, 0], 0, 0))
                        self.obs_buffer.append(False)  # Don't save observation during return rotation

                        self.action_buffer.append(([-1 * turn, 0], 0, 0))
                        self.obs_buffer.append(False)  # Don't save observation during return rotation

                        return True
                    else:
                        return False

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[MOVE_OBSTACLE] Failed after {max_retries} attempts: {str(e)}")

        return False

    def _process_search_move_result(self):
        prompt = search_move_prompt()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.around_image))
                print(f"[SEARCH-MOVE] \n {res} \n\n\n")
                pattern = re.compile(r'<think>(.*?)</think>\s*<output>(.*?)</output>', re.DOTALL)
                match = pattern.search(res)

                if match:
                    output = match.group(2).strip()

                    # Queue appropriate rotation actions
                    if output == "left":
                        for _ in range(3):
                            self.action_buffer.append(([-30, 0], 0, 0))
                    elif output == "right":
                        for _ in range(3):
                            self.action_buffer.append(([30, 0], 0, 0))

                    # Add forward actions
                    for _ in range(3):
                        self.action_buffer.append(([0, 100], 0, 0))

                    # Add a search action if not carrying
                    if self.info["picked"] != 1:
                        self.action_buffer.append(([0, 50], 0, 1))

                    return True

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[SEARCH_MOVE] Failed after {max_retries} attempts: {str(e)}")

        return False

    # Original helper methods preserved
    def analyse_initial_clue(self):
        # res0 = call_api_vlm(prompt=in_out_door_prompt(self.clue), base64_image=self.image_clue)
        prompt = initial_clue_prompt(self.clue)
        # prompt = initial_clue_prompt(self.clue)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = call_api_llm(prompt=prompt)
                print(res)
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>\s*<c>(.*?)</c>', re.DOTALL)
                match = pattern.search(res)

                if match:
                    direction = match.group(2).strip()
                    # Queue appropriate rotation actions based on direction
                    if direction == "left":
                        for _ in range(3):
                            self.action_buffer.append(([-30, 0], 0, 0))
                            self.obs_buffer.append(False)
                    elif direction == "right":
                        for _ in range(3):
                            self.action_buffer.append(([30, 0], 0, 0))
                            self.obs_buffer.append(False)
                    elif direction == "back":
                        for _ in range(6):
                            self.action_buffer.append(([30, 0], 0, 0))
                            self.obs_buffer.append(False)
                    elif "front" in direction and "left" in direction:
                        self.action_buffer.append(([-30, 0], 0, 0))
                        self.obs_buffer.append(False)
                    elif "front" in direction and "right" in direction:
                        self.action_buffer.append(([30, 0], 0, 0))
                        self.obs_buffer.append(False)
                    elif "rear" in direction and "left" in direction:
                        for _ in range(4):
                            self.action_buffer.append(([-30, 0], 0, 0))
                            self.obs_buffer.append(False)
                    elif "rear" in direction and "right" in direction:
                        for _ in range(4):
                            self.action_buffer.append(([30, 0], 0, 0))
                            self.obs_buffer.append(False)
                    # if res0 == "0":
                    #    for _ in range(10):
                    #        self.action_buffer.append(([0, 100], 0, 0))
                    # self.action_buffer.append(([0, 0], 0, 0))
                    # self.obs_buffer.append(True)
                    self.action_buffer.append(([0, 0], 0, 0))
                    self.obs_buffer.append(True)

                    # Parse landmarks
                    # self.landmark = ["ambulance"]
                    # if res0 == "0":
                    #    self.landmark = ["door"]
                    self.landmark += match.group(3).strip().split(",")
                    self.landmark = [
                        "injured person" if landmark.strip().lower() == "injured woman" or landmark.strip().lower() == "injured man"
                        else landmark.strip() for landmark in self.landmark]
                    self.landmark = [landmark for landmark in self.landmark if
                                     "ench" not in landmark and "mbulance" not in landmark]
                    # if res0 == "0":
                    #    self.landmark_back = ['door', 'ambulance', 'stretcher']
                    # else:
                    self.landmark_back = ['ambulance', 'orange bench']
                    return

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[CLUE ANALYSE] Failed after {max_retries} attempts: {str(e)}")

    def analyse_initial_image(self):
        prompt = initial_image_prompt()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.image_clue))
                self.person_text = res
                print(res)
                return True

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[IMAGE ANALYSE] Failed after {max_retries} attempts: {str(e)}")

        return False

    def encode_image_array(self, image_array):
        # Convert the image array to a PIL Image object
        image = Image.fromarray(np.uint8(image_array))

        # Save the PIL Image object to a bytes buffer
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")

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

        # Create a copy to avoid modifying the original
        image_copy = image_array.copy()
        cv2.line(image_copy, (line1, 0), (line1, h), line_color, line_thickness)
        cv2.line(image_copy, (line2, 0), (line2, h), line_color, line_thickness)

        return image_copy



    def reset(self, text, image):
        self.clue = text
        self.image_clue = image
        return


