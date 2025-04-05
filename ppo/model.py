import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from ppo.utils import debug_print

class CNN(nn.Module):
    def __init__(self, input_channels=3, output_dim=512):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.ReLU()
        )
        
        # Calculate output size
        self.mlp = nn.Sequential(
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        )
        
    def forward(self, x):
        x = x / 255.0
        # debug_print(x[0])
        x = self.cnn(x)
        # print('x shape:',x.shape)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x

class Actor(nn.Module):
    def __init__(self, state_dim=512, args=None):
        super(Actor, self).__init__()
        self.cnn = CNN(output_dim=args.latent_dim)
        
        # For continuous actions (angular, velocity)
        self.mean_angular = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.mean_velocity = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(2))
        # For discrete actions (viewport, interaction)
        self.viewport = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # None, Look Up, Look Down
        )
        self.interaction = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # None, Rescue, Place, etc
        )
    
    def normalize_action(self, action):
        # Normalize action to [-1, 1]
        action[:, 0] = action[:, 0] / 30.0
        action[:, 1] = action[:, 1] / 100.0
        return action

    def unnormalize_action(self, action):
        # Unnormalize action from [-1, 1] to original scale
        action[:, 0] = action[:, 0] * 30.0
        action[:, 1] = action[:, 1] * 100.0
        return action
        
    def get_action(self, state, deterministic=False):
        features = self.cnn(state)
        
        # Continuous actions
        mean_angular = self.mean_angular(features)
        mean_velocity = self.mean_velocity(features)
        means = torch.cat([mean_angular, mean_velocity], dim=-1)
        
        if deterministic:
            continuous_actions = means
        else:
            std = self.log_std.exp()
            dist = Normal(means, std)
            continuous_actions = dist.sample()
        
        # Discrete actions
        viewport_logits = self.viewport(features)
        interaction_logits = self.interaction(features)
        
        if deterministic:
            viewport_action = torch.argmax(viewport_logits, dim=-1)
            interaction_action = torch.argmax(interaction_logits, dim=-1)
        else:
            viewport_dist = Categorical(logits=viewport_logits)
            interaction_dist = Categorical(logits=interaction_logits)
            viewport_action = viewport_dist.sample()
            interaction_action = interaction_dist.sample()
        # print(continuous_actions)
            
        return continuous_actions, viewport_action, interaction_action
    
    def evaluate_actions(self, state, continuous_actions, viewport_action, interaction_action):
        #continuous_actions = self.normalize_action(continuous_actions)
        features = self.cnn(state)
        
        # Evaluate continuous actions
        mean_angular = self.mean_angular(features)
        mean_velocity = self.mean_velocity(features)
        means = torch.cat([mean_angular, mean_velocity], dim=-1)
        std = self.log_std.exp()
        dist = Normal(means, std)
        # print(continuous_actions.shape, means.shape, std.shape, features.shape, state.shape)
        continuous_log_probs = dist.log_prob(continuous_actions).sum(-1, keepdim=True)
        
        # Evaluate discrete actions
        viewport_logits = self.viewport(features)
        interaction_logits = self.interaction(features)
        viewport_dist = Categorical(logits=viewport_logits)
        interaction_dist = Categorical(logits=interaction_logits)
        
        viewport_log_probs = viewport_dist.log_prob(viewport_action).unsqueeze(-1)
        interaction_log_probs = interaction_dist.log_prob(interaction_action).unsqueeze(-1)
        
        # Combine log probs
        log_probs = continuous_log_probs + viewport_log_probs + interaction_log_probs
        
        # Calculate entropy
        entropy = dist.entropy().mean() + viewport_dist.entropy().mean() + interaction_dist.entropy().mean()
        
        # print(log_probs.shape, entropy.shape)
        
        return log_probs.squeeze(-1), entropy

class Critic(nn.Module):
    def __init__(self, state_dim, args=None):
        super(Critic, self).__init__()
        self.cnn = CNN(output_dim=args.latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + args.latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, state, additional_features):
        cnn_features = self.cnn(state)
        x = torch.cat([cnn_features, additional_features], dim=-1)
        x = self.mlp(x)
        return x.squeeze(-1)
