import time
import sys
import json
import cv2
import numpy as np
import os
from io import BytesIO
import base64
from ultralytics import YOLO
# 加载预训练的YOLO模型，如果下载了特定版本的权重，请指定路径

class AlgSolution:

    def __init__(self):
        self.yolo_model = YOLO('checkpoints/yolo11x.pt')  #  'yolov8n.pt' is the pretrained model file name for YOLOv8 nano version, replace it according to the actual situation
        if os.path.exists('/home/admin/workspace/job/logs/'):
            self.handle = open('/home/admin/workspace/job/logs/user.log', 'w')
        else:
            self.handle = open('user.log', 'w')
        self.handle.write("model loaded\n")
        self.handle.flush()
        self.foreward = {
            'angular': 0, # [-30, 30]
            'velocity': 50, # [-100, 100],
            'viewport': 0, # {0: stay, 1: look up, 2: look down},
            'interaction': 0, #
        }
        self.backward = {
            'angular': 0, # [-30, 30]
            'velocity': -50, # [-100, 100],
            'viewport': 0, # {0: stay, 1: look up, 2: look down},
            'interaction': 0, #
        }
        self.turnleft = {
            'angular': -20, # [-30, 30]
            'velocity': 0, # [-100, 100],
            'viewport': 0, # {0: stay, 1: look up, 2: look down},
            'interaction': 0, #
        }
        self.turnright = {
            'angular': 20, # [-30, 30]
            'velocity': 0, # [-100, 100],
            'viewport': 0, # {0: stay, 1: look up, 2: look down},
            'interaction': 0, #
        }
        self.carry = {
            'angular': 0, # [-30, 30]
            'velocity': 0, # [-100, 100],
            'viewport': 0, # {0: keep, 1: up, 2: down},
            'interaction': 3,
        }
        self.drop = {
            'angular': 0, # [-30, 30]
            'velocity': 0, # [-100, 100],
            'viewport': 0, # {0: keep, 1: up, 2: down},
            'interaction': 4,
        }
        self.noaction = {
            'angular': 0, # [-30, 30]
            'velocity': 0, # [-100, 100],
            'viewport': 0, # {0: stay, 1: look up, 2: look down},
            'interaction': 0, # {0: stand, 1: jump, 2: crouch, 3: carry, 4: drop, 5: open door}
        }

        self.person_success = False
        self.right_times = 0
        self.foreward_times = 0
        self.backward_times = 0
        self.try_times = 10
        self.idx = 0

    def reset(self, reference_text=None, reference_image=None):
        self.person_success = False
        self.right_times = 0
        self.foreward_times = 0
        self.backward_times = 0
        self.try_times = 10
        self.idx = 0

    def predicts(self, ob,success):
        # person_exist = False
        # truck_exist = False
        self.handle.write('Step %d\n'%self.idx)
        self.handle.flush()
        self.idx += 1
        if self.idx == 200:
            return self.drop

        ob = base64.b64decode(ob)
        ob = cv2.imdecode(np.frombuffer(ob, np.uint8), cv2.IMREAD_COLOR)
        results = self.yolo_model(source=ob,imgsz=640,conf=0.1) 
        boxes = results[0].boxes  # get all detected bounding box
        if boxes.shape[0] == 0:
            # randomly choose a number from list[1,2]
            if self.right_times >= self.try_times:
                if self.foreward_times<self.try_times:
                    action = self.foreward
                    time.sleep(2)
                    self.handle.write(" ---------------- no object foreward ------------------\n")
                    self.foreward_times += 1
                    self.handle.write(json.dumps(action) + '\n')
                    self.handle.flush()
                    return action
                else:
                    if self.backward_times<self.try_times:
                        action = self.backward
                        time.sleep(2)
                        self.handle.write(" ---------------- no object backward ------------------\n")
                        self.backward_times += 1
                        self.handle.write(json.dumps(action) + '\n')
                        self.handle.flush()
                        return action
            else:
                action = self.turnright
                time.sleep(2)
                self.handle.write(" ---------------- no object turnright ------------------\n")
                self.right_times += 1
                self.handle.write(json.dumps(action) + '\n')
                self.handle.flush()
                return action
            if self.right_times >= self.try_times and self.foreward_times>=self.try_times and self.backward_times>=self.try_times:
                #self.right_times = 9
                self.right_times = 9
                self.foreward_times = 0
                self.backward_times= 0
                action = self.turnright
                time.sleep(2)
                self.handle.write(" ---------------- no object turnright ------------------\n")
                self.handle.write(json.dumps(action) + '\n')
                self.handle.flush()
                return action
        for box in boxes:
            cls = int(box.cls.item())  # get class ID
            if self.yolo_model.names[cls] == 'person' and self.person_success is False:  # 检查是否是person类别
                # get bounding box coordinate（x1, y1, x2, y2）
                #x0, y0, w_, h_ = box.xywh
                res_ = box.xywh
                self.handle.write("res_person: %s\n"%res_)
                self.handle.flush()

                x0, y0, w_, h_ = res_[0].tolist()
                self.handle.write("x0, y0, w_, h_: %s, %s, %s, %s\n"%(x0, y0, w_, h_))
                self.handle.flush()
                if w_ > h_:
                    if y0-0.5*h_ > 400:
                        action = self.carry 
                        time.sleep(2)
                        self.handle.write("carry ！！！！！！！！！！！！！！！！！！\n")
                        self.person_success = True
                        self.handle.write(json.dumps(action) + '\n')
                        self.handle.flush()
                        return action
                    else:
                        if x0 < 220:
                            action = self.turnleft
                            time.sleep(2)
                            self.handle.write(" ---------------- turnleft ------------------\n")
                            self.handle.write(json.dumps(action) + '\n')
                            self.handle.flush()
                            return action
                        elif x0 > 420:
                            action = self.turnright
                            time.sleep(2)
                            self.handle.write(" ---------------- turnright ------------------\n")
                            self.handle.write(json.dumps(action) + '\n')
                            self.handle.flush()
                            return action
                        else:
                            action = self.foreward
                            time.sleep(2)
                            self.handle.write(" ---------------- foreward ------------------\n")
                            self.handle.write(json.dumps(action) + '\n')
                            self.handle.flush()
                            return action
            elif self.yolo_model.names[cls] == 'truck' and self.person_success:  #  Check if it is the 'person' category
                # get bounding box coordinate（x1, y1, x2, y2）
                #x0, y0, w_, h_ = box.xywh
                res_ = box.xywh
                self.handle.write("res_truck: %s\n"%res_)
                self.handle.flush()

                x0, y0, w_, h_ = res_[0].tolist()
                if w_ > 390:
                    action = self.drop 
                    time.sleep(2)
                    self.handle.write(" ============================================= drop =============================================\n")
                    self.handle.write(json.dumps(action) + '\n')
                    self.handle.flush()
                    return action
                else:
                    if x0 < 220:
                        action = self.turnleft
                        time.sleep(2)
                        self.handle.write(" =============== turnleft ===============\n")
                        self.handle.write(json.dumps(action) + '\n')
                        self.handle.flush()
                        return action
                    elif x0 > 420:
                        action = self.turnright
                        self.handle.write(" =============== turnright ===============\n")
                        time.sleep(2)
                        self.handle.write(json.dumps(action) + '\n')
                        self.handle.flush()
                        return action
                    else:
                        action = self.foreward
                        time.sleep(2)
                        self.handle.write(" =============== forward ===============\n")
                        self.handle.write(json.dumps(action) + '\n')
                        self.handle.flush()
                        return action
        
        if self.right_times >= self.try_times:
            if self.foreward_times<self.try_times:
                action = self.foreward
                time.sleep(2)
                self.handle.write(" ---------------- no person and truck foreward------------------\n")
                self.foreward_times += 1
                self.handle.write(json.dumps(action) + '\n')
                self.handle.flush()
                return action
            else:
                if self.backward_times<self.try_times:
                    action = self.backward
                    time.sleep(2)
                    self.handle.write(" ---------------- no person and truck backward------------------\n")
                    self.backward_times += 1
                    self.handle.write(json.dumps(action) + '\n')
                    self.handle.flush()
                    return action
        else:
            action = self.turnright
            time.sleep(2)
            self.handle.write(" ---------------- no person and truck turnright------------------\n")
            self.right_times += 1
            self.handle.write(json.dumps(action) + '\n')
            self.handle.flush()
            return action

        if self.right_times >= self.try_times and self.foreward_times>=self.try_times and self.backward_times>=self.try_times:
            #self.right_times = 9
            self.right_times = 9
            self.foreward_times = 0
            self.backward_times= 0
            action = self.turnright
            time.sleep(2)
            self.handle.write(" ---------------- no person and truck turnright------------------\n")
            self.handle.write(json.dumps(action) + '\n')
            self.handle.flush()
            return action
        else:
            action = self.noaction
            time.sleep(2)
            self.handle.write(" ---------------- no person and truck turnright------------------\n")
            self.handle.write(json.dumps(action) + '\n')
            self.handle.flush()
            return action

