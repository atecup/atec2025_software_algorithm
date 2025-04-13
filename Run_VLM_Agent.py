import gym_rescue
from gym_rescue.envs.wrappers import switch_env, configUE, early_done
import gym
import numpy as np
import cv2
import os
import json
import base64
import time
import traceback

from VLM_Agent.solution_yoloVLM import AlgSolution

UnrealEnv = os.environ.get('UnrealEnv')
TEST_JSONL = os.path.join(UnrealEnv, 'test_L1.jsonl')

def img2base64(img):
    buf = cv2.imencode('.png', img)[1]
    b64 = base64.b64encode(buf).decode('utf-8')
    return b64


agent = AlgSolution()

results = []
seed = 0

rewards = []
ep_times = []

env_old = None
with open(TEST_JSONL, "r") as fp:
    for line in fp.readlines():

        content = json.loads(line)
        level = content.get('level')
        env_id = content.get("env_id")
        agent_loc = content.get("agent_loc")
        injured_player_loc = content.get("injured_player_loc")
        injured_agent_id = content.get("injured_agent_id")
        stretcher_loc = content.get("stretcher_loc")
        ambulance_loc= content.get("ambulance_loc")
        reference_text = content.get("reference_text")
        reference_image_path = content.get("reference_image_path")
        max_steps = 9999999
        timeout = content.get("timeout")

        env = gym.make(env_id, action_type='Mixed', observation_type='Color', reset_type=level)
        if env_old is not None:
            env = switch_env.SwitchEnvWrapper(env, env_old)
        env_old = env
        try:
            env._max_episode_steps = max_steps
            env = configUE.ConfigUEWrapper(env, offscreen=True, resolution=(640,480))
            env = early_done.EarlyDoneWrapper(env, int(timeout))

            if reference_image_path:
                reference_image_path = os.path.join(UnrealEnv, 'ref_image', reference_image_path[0])
                reference_image = base64.b64encode(open(reference_image_path, 'rb').read()).decode("utf-8")
            else:
                reference_image = None

            agent.reset(reference_text, reference_image)

            env.seed(seed)
            seed += 1

            env.unwrapped.injured_player_pose = injured_player_loc
            env.unwrapped.rescue_pose = stretcher_loc
            env.unwrapped.agent_pose = agent_loc
            env.unwrapped.ambulance_pose = ambulance_loc
            ob, info = env.reset()
            env.unwrapped.unrealcv.set_appearance(env.unwrapped.injured_agent, injured_agent_id)

            ob, reward, termination, truncation, info = env.step([[(0, 0), 0, 0]])
            current_step = 0
            picked_reward = 0.
            success = True

            start_t = time.time()
            while True:
                pred = agent.predicts(
                    img2base64(ob),
                    success,
                )
                action = [(pred['angular'], pred['velocity']), pred['viewport'], pred['interaction']]
                ob, reward, termination, truncation, info = env.step([action])
                if info['picked']:
                    picked_reward = 0.5

                if pred['interaction'] == 3:
                    if info['picked']:
                        success = True
                    else:
                        success = False
                else:
                    success = True

                if current_step % 1 == 0:
                    print('step', current_step)
                end_t = time.time()
                if termination:
                    rewards.append(1)
                    ep_times.append(end_t - start_t)
                    break
                if truncation or pred['interaction'] == 4:
                    rewards.append(picked_reward)
                    ep_times.append(end_t - start_t)
                    break

                current_step += 1
        except:
            env.close()
            traceback.print_exc()
            exit()

env.close()

print('Rewards', rewards)
print('Times', ep_times)
rewards = np.mean(rewards)
ep_times = np.sum(ep_times)
print('Rewards', rewards, 'Times', ep_times)

