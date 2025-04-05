import torch
import torch.nn as nn
import numpy as np
from ppo.valuenorm import ValueNorm
from ppo.utils import debug_print
import torch.nn.functional as F

class PPO:
    def __init__(self, actor, critic, actor_optimizer, critic_optimizer, args):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        
        self.clip_param = args.clip_param
        self.max_grad_norm = args.max_grad_norm
        self.entropy_coef = args.entropy_coef
        self.value_loss_coef = args.value_loss_coef
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.device = args.device
        
        # Add ValueNorm
        self.value_normalizer = ValueNorm(1, beta=0.999, device=self.device)
        
        # Add new parameters
        self.use_huber_loss = getattr(args, 'use_huber_loss', True)
        self.huber_delta = getattr(args, 'huber_delta', 10.0)
        self.use_clipped_value_loss = getattr(args, 'use_clipped_value_loss', False)
        
    def compute_gae(self, values, rewards, masks):
        advantages = torch.zeros_like(rewards)
        gae = 0
        denormed_values = torch.tensor(self.value_normalizer.denormalize(values.detach().cpu().numpy()), device=self.device)
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = denormed_values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * masks[t] - denormed_values[t]
            gae = delta + self.gamma * self.gae_lambda * masks[t] * gae
            # debug_print(gae.shape, advantages.shape, rewards.shape, values.shape, masks.shape)
            advantages[t] = gae
            
        returns = advantages + denormed_values
        # debug_print('advantages.shape', advantages.shape, 'returns.shape', returns.shape, 'rewards.shape', rewards.shape, 'values.shape', values.shape, 'masks.shape', masks.shape)
        return advantages, returns
        
    def pg_loss(self, sample):
        share_obs, obs, returns, continuous_actions, viewport_actions, interaction_actions, \
        old_log_probs, masks, values = sample
        
        # Update value normalizer with returns
        # debug_print(returns.shape)
        # self.value_normalizer.update(returns.unsqueeze(-1))
        
        # Calculate advantages using normalized values
        normalized_values = self.value_normalizer.normalize(values.unsqueeze(-1)).squeeze(-1)
        # debug_print(normalized_values, self.value_normalizer.running_mean, self.value_normalizer.running_mean_sq, self.value_normalizer.debiasing_term)
        advantages, returns = self.compute_gae(normalized_values, returns, masks)
        advantages = (advantages) / (advantages.std() + 1e-8)
        
        # Get new log probs and entropy
        # print(obs.shape)
        log_probs, entropy = self.actor.evaluate_actions(
            obs.reshape(-1, *obs.shape[2:]), continuous_actions.reshape(-1, *continuous_actions.shape[2:]), viewport_actions.reshape(-1, *viewport_actions.shape[2:]), interaction_actions.reshape(-1, *interaction_actions.shape[2:])
        )
        log_probs = log_probs.reshape(*obs.shape[:2])
        
        # Calculate policy loss
        # debug_print(log_probs.shape, old_log_probs.shape, advantages.shape)
        ratio = torch.exp(log_probs - old_log_probs)
        # debug_print(log_probs.shape, old_log_probs.shape, advantages.shape, ratio.shape)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Calculate value loss with normalization and clipping
        new_values = self.critic(share_obs.reshape(-1, *share_obs.shape[2:]), 
                               torch.zeros((share_obs.shape[0] * share_obs.shape[1], 0)).to(self.device))
        new_values = new_values.reshape(returns.shape)
        
        # print('values', self.value_normalizer.denormalize(values.detach().cpu().numpy()).mean())
        # print('new_values', self.value_normalizer.denormalize(new_values.detach().cpu().numpy()).mean())
        # print('returns', returns.detach().cpu().numpy().mean())
        
        # Clip value predictions
        value_pred_clipped = values + (new_values - values).clamp(-self.clip_param, self.clip_param)
        
        # Normalize returns
        normalized_returns = self.value_normalizer.normalize(returns.unsqueeze(-1)).squeeze(-1)
        
        # Calculate errors
        error_clipped = normalized_returns - value_pred_clipped
        error_original = normalized_returns - new_values
        
        # Calculate losses using either Huber or MSE
        if self.use_huber_loss:
            value_loss_clipped = F.huber_loss(value_pred_clipped, normalized_returns, delta=self.huber_delta, reduction='none')
            value_loss_original = F.huber_loss(new_values, normalized_returns, delta=self.huber_delta, reduction='none')
        else:
            value_loss_clipped = F.mse_loss(value_pred_clipped, normalized_returns, reduction='none')
            value_loss_original = F.mse_loss(new_values, normalized_returns, reduction='none')
        
        # Use clipped loss if enabled
        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_clipped, value_loss_original).mean()
        else:
            value_loss = value_loss_original.mean()
        
        # Update networks
        self.actor_optimizer.zero_grad()
        (policy_loss - entropy * self.entropy_coef).backward()
        
        # Log actor grad norm before clipping
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float('inf'))
        # print('Actor grad norm before clipping:', actor_grad_norm.item())
        
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        
        # Log critic grad norm before clipping
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float('inf'))
        # print('Critic grad norm before clipping:', critic_grad_norm.item())
        
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        return value_loss.item(), policy_loss.item(), entropy.item(), ratio.mean().item()