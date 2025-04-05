import time
import torch
import torch.optim as optim
import numpy as np
import gym
import gym_rescue
from gym_rescue.envs.wrappers import configUE,early_done
import os
import argparse
import json
import base64
import cv2
from collections import defaultdict
from ppo.utils import debug_print
from ppo.model import Actor, Critic
from ppo.ppo import PPO
from ppo.runner import Runner
from ppo.multi_process_wrapper import MultiEnvWrapper

class VideoLogger:
    def __init__(self, video_dir, fps=30):
        self.video_dir = video_dir
        self.fps = fps
        os.makedirs(video_dir, exist_ok=True)
        self.env_frames = defaultdict(list)
        
    def add_frame(self, env_idx, frame):
        self.env_frames[env_idx].append(frame)
    
    def save_videos(self, epoch):
        for env_idx, frames in self.env_frames.items():
            if not frames:
                continue
                
            video_path = os.path.join(self.video_dir, f"epoch_{epoch}_env_{env_idx}.mp4")
            height, width = frames[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
            
            for frame in frames:
                out.write(frame)
            out.release()
            
        # Clear frames after saving
        self.env_frames.clear()

def parse_args():
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--dataset_root", type=str, default=os.environ.get('DATASET_ROOT'))
    parser.add_argument("--test_jsonl", type=str, default='debug_test.jsonl')
    parser.add_argument("--latent_dim", type=int, default=512)
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--num_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-3)
    
    # PPO specific
    parser.add_argument("--ppo_epochs", type=int, default=20, 
        help="Number of times to optimize on the same batch of data")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    
    # Other
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--load_path", type=str, default="checkpoints/model_epoch.pt")
    # Video logging
    parser.add_argument("--save_video", action="store_true", help="Save training videos")
    parser.add_argument("--video_dir", type=str, default="videos")
    parser.add_argument("--video_interval", type=int, default=10, help="Save video every N epochs")
    parser.add_argument("--fps", type=int, default=30)
    
    args = parser.parse_args()
    args.test_jsonl = os.path.join(args.dataset_root, args.test_jsonl)
    return args

def load_env_configs(jsonl_path):
    configs = []
    with open(jsonl_path, "r") as fp:
        for line in fp.readlines():
            config = json.loads(line)
            # Ensure required fields exist
            if 'env_id' not in config:
                config['env_id'] = 'UnrealRescue-track_train'
            if 'max_steps' not in config:
                config['max_steps'] = 2000
            configs.append(config)
    return configs

def make_env(env_configs, env_idx):

    def _init():
        # import pdb
        # pdb.set_trace()
        config = env_configs[env_idx % len(env_configs)]
        env = gym.make(
            config['env_id'],
            action_type='Mixed',
            observation_type='Color',
            reset_type=config['level']
        )
        env._max_episode_steps = config['max_steps']
        env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=(240,240))
        env=early_done.EarlyDoneWrapper(env, 300)

        env.unwrapped.injured_player_pose = config['injured_player_loc']
        env.unwrapped.rescue_pose = config['stretcher_loc']
        env.unwrapped.agent_pose = config['agent_loc']
        env.unwrapped.ambulance_pose = config['ambulance_loc']
        # from gym_rescue.envs.wrappers import time_dilation
        # env = time_dilation.TimeDilationWrapper(env, reference_fps=120, update_steps=10) 
        
        # Store config in env for use during reset
        return env

    return _init

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_video:
        video_logger = VideoLogger(args.video_dir, args.fps)
    
    # Load environment configurations
    env_configs = load_env_configs(args.test_jsonl)
    
    # Initialize environments with multiprocessing
    env_fns = [make_env(env_configs, i) for i in range(args.num_envs)]
    print('Creating MultiEnvWrapper')
    envs = MultiEnvWrapper(env_fns)
    print('MultiEnvWrapper created')
    debug_print('OK')
    # Get observation and action spaces from first environment
    # This is needed since we're using multiprocessing
    observation_space = envs.observation_space
    action_space = envs.action_space
    
    # Initialize networks with correct dimensions
    actor = Actor(
        state_dim=args.latent_dim,
        # obs_space=observation_space,
        # action_space=action_space,
        args=args
    ).to(args.device)
    
    critic = Critic(
        state_dim=0,
        # obs_space=observation_space,
        args=args
    ).to(args.device)
    
    # Initialize optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)
    
    # Initialize PPO agent
    agent = PPO(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        args=args
    )
    
    # Initialize runner
    runner = Runner(envs, agent, args)
    # time.sleep(10)
    
    # Training loop
    total_steps = 0
    for epoch in range(args.num_epochs):
        # Collect experience
        start_time = time.time()
        batch = runner.run()
        end_time = time.time()
        print(f"Time taken for one epoch: {end_time - start_time} seconds, FPS: {args.num_steps * args.num_envs / (end_time - start_time)}")
        
        # If video saving is enabled and it's a video interval epoch
        if args.save_video and (epoch + 1) % args.video_interval == 0:
            # Get observations from the runner
            obs_tensor = batch[0]  # Shape: (num_steps, num_envs, C, H, W)
            
            # Process each environment's observations
            for env_idx in range(args.num_envs):
                env_frames = obs_tensor[:, env_idx].cpu().numpy()
                # Convert from (T, C, H, W) to (T, H, W, C)
                env_frames = np.transpose(env_frames, (0, 2, 3, 1))
                # Convert to uint8 if necessary
                if env_frames.dtype != np.uint8:
                    env_frames = (env_frames * 255).astype(np.uint8)
                # Add frames to video logger
                for frame in env_frames:
                    video_logger.add_frame(env_idx, frame)
            
            # Save videos for this epoch
            video_logger.save_videos(epoch)
        
        # Multiple PPO updates on the same batch
        total_value_loss = 0
        total_policy_loss = 0
        total_entropy = 0
        total_ratio = 0
        for _ in range(args.ppo_epochs):
            value_loss, policy_loss, entropy, ratio = agent.pg_loss(batch)
            total_value_loss += value_loss
            total_policy_loss += policy_loss
            total_entropy += entropy
            total_ratio += ratio
        # Average the losses
        avg_value_loss = total_value_loss / args.ppo_epochs
        avg_policy_loss = total_policy_loss / args.ppo_epochs
        avg_entropy = total_entropy / args.ppo_epochs
        avg_ratio = total_ratio / args.ppo_epochs
        total_steps += args.num_steps * args.num_envs
        
        # Logging
        print(f"Epoch {epoch}")
        print(f"Total Steps: {total_steps}")
        print(f"Average Value Loss: {avg_value_loss:.4f}")
        print(f"Average Policy Loss: {avg_policy_loss:.4f}")
        print(f"Average Entropy: {avg_entropy:.4f}")
        print(f"Average Ratio: {avg_ratio:.4f}")
        # Save model
        if (epoch + 1) % args.save_interval == 0:
            model_path = os.path.join(args.save_dir, f"model_epoch.pt")
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
            }, model_path)
            print(f"Saved model to {model_path}")
    
    # Clean up
    envs.close()

if __name__ == "__main__":
    main() 
