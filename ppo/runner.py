import time
import torch
import numpy as np
import cv2
import base64
from ppo.utils import debug_print

def img2base64(img):
    buf = cv2.imencode('.png', img)[1]
    b64 = base64.b64encode(buf).decode('utf-8')
    return b64

def normalize_action(action):
    action = action.clone()
    # Normalize action to [-1, 1]
    action[:, 0] = action[:, 0] / 10.0
    action[:, 1] = action[:, 1] / 10.0
    return action

def unnormalize_action(action):
    action = action.clone()
    # Unnormalize action from [-1, 1] to original scale
    action[:, 0] = action[:, 0] * 10.0
    action[:, 1] = action[:, 1] * 10.0
    return action
class Runner:
    def __init__(self, envs, actor_critic, args):
        self.envs = envs
        self.actor_critic = actor_critic
        self.num_envs = len(envs)
        self.device = args.device

        # Storage setup
        self.num_steps = args.num_steps
        self.obs = []
        self.share_obs = []
        self.rewards = []
        self.values = []
        self.returns = []
        self.continuous_actions = []
        self.viewport_actions = []
        self.interaction_actions = []
        self.action_log_probs = []
        self.masks = []

        # Initialize buffers
        self.reset_buffers()

    def reset_buffers(self):
        self.obs = []
        self.share_obs = []
        self.rewards = []
        self.values = []
        self.returns = []
        self.continuous_actions = []
        self.viewport_actions = []
        self.interaction_actions = []
        self.action_log_probs = []
        self.masks = []

    def run(self):
        # Reset environments using the multi-env wrapper.
        # Assume the returned obs shape is (num_envs, H, W, C).

        obs, _ = self.envs.reset()

        # print(obs.shape)
        tot_time = 0
        # Convert observations to tensor with shape (num_envs, C, H, W)
        # obs=np.expand_dims(obs, axis=0)
        debug_print(obs.shape)

        obs_tensor = torch.FloatTensor(np.transpose(obs, (0, 3, 1, 2))).to(self.device)
        for step in range(self.num_steps):
            # Get actions using the original three-output logic.
            with torch.no_grad():

                continuous_actions, viewport_actions, interaction_actions = self.actor_critic.actor.get_action(obs_tensor)
                # print(continuous_actions.shape)
                values = self.actor_critic.critic(obs_tensor, torch.zeros((self.num_envs, 0)).to(self.device))
                action_log_probs = self.actor_critic.actor.evaluate_actions(
                    obs_tensor, continuous_actions, viewport_actions, interaction_actions
                )[0]

            # Convert actions to environment format per environment.
            env_actions = []
            for i in range(self.num_envs):
                tmp = unnormalize_action(continuous_actions)
                action = [
                    (tmp[i][0].item(), tmp[i][1].item()),
                    viewport_actions[i].item(),
                    interaction_actions[i].item()
                ]
                env_actions.append([action])

            # Step environments via the multi-env wrapper.
            start_time = time.time()
            next_obs, rewards, terminations, truncations, infos = self.envs.step(env_actions)
            # next_obs = np.expand_dims(next_obs, axis=0)

            rewards = np.clip(rewards, -10, 10)
            end_time = time.time()
            tot_time += end_time - start_time
            # Convert next observations: each from (H, W, C) to (C, H, W)
            # debug_print(next_obs.shape)

            next_obs_tensor = torch.FloatTensor(np.transpose(next_obs, (0, 3, 1, 2))).to(self.device)

            # Process rewards and masks.
            rewards_processed = []
            masks = []
            for i in range(self.num_envs):
                r = rewards[i] #/ 10.0
                # Adjust reward if the 'picked' flag is set.
                if infos[i].get('picked', False):
                    r = 1
                rewards_processed.append(r)
                done = terminations[i] or truncations[i]
                masks.append(0.0 if done else 1.0)

            # Store transition.
            self.obs.append(obs_tensor)
            # If share_obs isn't separately maintained, mirror obs.
            self.share_obs.append(obs_tensor)
            self.values.append(values)
            self.rewards.append(torch.FloatTensor(rewards_processed).to(self.device))
            self.continuous_actions.append(continuous_actions)
            self.viewport_actions.append(viewport_actions)
            self.interaction_actions.append(interaction_actions)
            self.action_log_probs.append(action_log_probs)
            self.masks.append(torch.FloatTensor(masks).to(self.device))

            # Update obs_tensor for next step.
            obs_tensor = next_obs_tensor

            # Break if all environments are done.
            if all([t or tr for t, tr in zip(terminations, truncations)]):
                break

        # Prepare batch identical to the original.
        print(torch.stack(self.rewards).sum()/len(self.envs))
        debug_print(f'Time taken for one epoch: {tot_time} seconds, FPS: {self.num_steps * self.num_envs / tot_time}')

        batch = (
            torch.stack(self.obs),
            torch.stack(self.share_obs),
            torch.stack(self.rewards),
            torch.stack(self.continuous_actions),
            torch.stack(self.viewport_actions),
            torch.stack(self.interaction_actions),
            torch.stack(self.action_log_probs),
            torch.stack(self.masks),
            torch.stack(self.values)
        )

        self.reset_buffers()
        return batch