import multiprocessing as mp
from multiprocessing import Process, Pipe
import numpy as np
import torch
import gym
import time
import psutil
import os

from ppo.utils import debug_print

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, termination, truncation, info = env.step(data)
                remote.send((ob, reward, termination, truncation, info))
            elif cmd == 'reset':
                ob, info = env.reset()
                remote.send((ob, info))
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
        except Exception as e:
            print(f"Error in worker: {e}")
            break

class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x
    
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class MultiEnvWrapper:
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        
        # Create pipes for each environment
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        
        # Get available CPU cores and their usage
        cpu_usages = psutil.cpu_percent(percpu=True)
        least_used_cpus = sorted(range(len(cpu_usages)), key=lambda i: cpu_usages[i])
        
        # Create processes
        self.processes = []
        for work_remote, remote, env_fn, cpu_id in zip(
            self.work_remotes, 
            self.remotes, 
            env_fns, 
            least_used_cpus * (len(env_fns) // len(least_used_cpus) + 1)  # Cycle through CPUs if more envs than cores
        ):
            process = Process(
                target=worker_with_affinity,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
                kwargs={'cpu': cpu_id}
            )
            process.daemon = True
            process.start()
            self.processes.append(process)
            work_remote.close()
            time.sleep(10)
        
        # Get spaces from first environment
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space
    
    def __len__(self):
        return self.num_envs
    
    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
    
    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rewards, terminations, truncations, infos = zip(*results)
        # debug_print(rewards, terminations, truncations, obs[0].shape)
        return np.stack(obs, axis=0), np.stack(rewards, axis=0), np.stack(terminations, axis=0), np.stack(truncations, axis=0), infos
    
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
    
    def reset(self):
        # print("Resetting environments")
        for remote in self.remotes:
            remote.send(('reset', None))
        # print("Receiving results")
        # i = 0
        # for remote in self.remotes:
        #     print(f"Receiving result {i}")
        #     result = remote.recv()
        #     print(f"Result {i} received: {result}")
        #     i += 1
        results = [remote.recv() for remote in self.remotes]
        # print("Results received")
        # import pdb
        # pdb.set_trace()
        obs, infos = zip(*results)
        # return np.concatenate(obs, axis=0), infos

        return np.array(obs), infos

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

def worker_with_affinity(remote, parent_remote, env_fn_wrapper, cpu=None):
    parent_remote.close()
    env = env_fn_wrapper.x()
    obs, info = env.reset()
    print("C Reset complete")
    
    # Set CPU affinity if specified
    if cpu is not None:
        try:
            import psutil
            psutil.Process().cpu_affinity([cpu])
        except:
            print(f"Warning: Could not set CPU affinity to {cpu}")
    
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, reward, termination, truncation, info = env.step(data)
                remote.send((obs, reward, termination, truncation, info))
            elif cmd == 'reset':
                # print("Resetting environment")
                obs, info = env.reset()
                # print("Reset complete")
                remote.send((obs, info))
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command {cmd}")
        except EOFError:
            break

# def make_env(env_configs, env_idx):
#     def _init():
#         # Get config for this environment instance
#         config = env_configs[env_idx % len(env_configs)]
        
#         # Create the environment
#         env = gym.make(
#             config['env_id'],
#             action_type='Mixed',
#             observation_type='Color',
#             reset_type=config['level']
#         )
        
#         # Set max steps from config
#         env._max_episode_steps = config['max_steps']
        
#         # Add UE wrapper with offscreen rendering and resolution
#         env = configUE.ConfigUEWrapper(
#             offscreen=True,
#             resolution=(640, 480)
#         )
        
#         # Store config in env for use during reset
#         env.config = config
        
#         # Set specific locations if provided in config
#         if hasattr(env.unwrapped, 'rescue_pose') and 'stretcher_loc' in config:
#             env.unwrapped.rescue_pose = config['stretcher_loc']
            
#         if hasattr(env.unwrapped, 'injured_player_pose') and 'injured_player_loc' in config:
#             env.unwrapped.injured_player_pose = config['injured_player_loc']
            
#         return env
#     return _init 