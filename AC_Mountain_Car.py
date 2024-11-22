import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
from collections import deque
from PIL import Image

class TrainingMetrics:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        self.critic_losses = []
        # Track individual weights instead of statistics
        self.actor_output_weights = []  # Will store final layer weights
    
    def moving_average(self, data, window=5):
        """Calculate moving average with the specified window"""
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def update(self, reward, length, critic_loss, actor_model):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.critic_losses.append(critic_loss)
        
        # Extract and store final layer weights
        actor_params = list(actor_model.actor.parameters())
        output_weights = actor_params[-2].detach().cpu().numpy().flatten()  # Get final layer weights
        self.actor_output_weights.append(output_weights)
        
        # Calculate success rate (last 100 episodes)
        successes = sum(1 for r in self.episode_rewards[-100:] if r >= 90)
        self.success_rate.append(successes / min(len(self.episode_rewards), 100))
    
    def plot(self, output_dir, episode):
        plt.style.use('default')
        fig = plt.figure(figsize=(15, 12))
        
        COLORS = {
            'reward': '#2ecc71',
            'length': '#e74c3c',
            'success': '#9b59b6',
            'loss': '#3498db',
        }
        ALPHA = 0.3
        SMOOTH_WINDOW = 5
        
        # Plot 1: Episode Rewards and Lengths
        ax1 = plt.subplot(2, 2, 1)
        episodes = np.arange(len(self.episode_rewards))
        
        ax1.plot(episodes, self.episode_rewards, color=COLORS['reward'], alpha=ALPHA, label='Raw Reward')
        ax1.plot(episodes, self.episode_lengths, color=COLORS['length'], alpha=ALPHA, label='Raw Length')
        
        if len(episodes) > SMOOTH_WINDOW:
            smooth_rewards = self.moving_average(self.episode_rewards, SMOOTH_WINDOW)
            smooth_lengths = self.moving_average(self.episode_lengths, SMOOTH_WINDOW)
            smooth_episodes = episodes[SMOOTH_WINDOW-1:]
            ax1.plot(smooth_episodes, smooth_rewards, color=COLORS['reward'], label=f'Reward (MA{SMOOTH_WINDOW})')
            ax1.plot(smooth_episodes, smooth_lengths, color=COLORS['length'], label=f'Length (MA{SMOOTH_WINDOW})')
        
        ax1.set_title('Episode Rewards and Lengths', fontsize=12)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Success Rate
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(episodes, self.success_rate, color=COLORS['success'], label='Success Rate')
        ax2.fill_between(episodes, self.success_rate, alpha=0.2, color=COLORS['success'])
        ax2.set_title('Success Rate (100-episode window)', fontsize=12)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Rate')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Loss Convergence
        ax3 = plt.subplot(2, 2, 3)
        steps = np.arange(len(self.critic_losses))
        
        ax3.plot(steps, self.critic_losses, color=COLORS['loss'], alpha=ALPHA, label='Raw Loss')
        if len(steps) > SMOOTH_WINDOW:
            smooth_loss = self.moving_average(self.critic_losses, SMOOTH_WINDOW)
            smooth_steps = steps[SMOOTH_WINDOW-1:]
            ax3.plot(smooth_steps, smooth_loss, color=COLORS['loss'], label=f'Loss (MA{SMOOTH_WINDOW})')
        
        ax3.set_title('Value Function Convergence', fontsize=12)
        ax3.set_xlabel('Update Step')
        ax3.set_ylabel('MSE Loss')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Individual Weight Trajectories
        ax4 = plt.subplot(2, 2, 4)
        weight_trajectories = np.array(self.actor_output_weights)
        num_weights = weight_trajectories.shape[1]
        
        # Plot each individual weight trajectory
        for i in range(min(num_weights, 10)):  # Plot up to 10 weights to avoid cluttering
            ax4.plot(episodes, weight_trajectories[:, i], 
                    alpha=0.6, label=f'Weight {i+1}')
        
        ax4.set_title('Output Layer Weights Over Episodes', fontsize=12)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Weight Value')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'training_metrics_episode_{episode}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

class AnimationRecorder:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.frames = []
        self.recording = False
    
    def start_recording(self):
        self.recording = True
        self.frames = []
    
    def stop_recording(self):
        self.recording = False
    
    def add_frame(self, env):
        """Capture a frame from the environment if recording"""
        if self.recording:
            frame = env.render()
            self.frames.append(Image.fromarray(frame))
    
    def save_gif(self, episode):
        """Save the recorded frames as a GIF"""
        if self.frames:
            path = os.path.join(self.output_dir, f'episode_{episode}.gif')
            self.frames[0].save(
                path,
                save_all=True,
                append_images=self.frames[1:],
                duration=50,
                loop=0
            )
            self.frames = []
            self.recording = False

class ActorCritic(nn.Module):
    def __init__(self, state_dim=2, action_dim=1, hidden_dim=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.414)
            module.bias.data.zero_()
    
    def get_action_dist(self, state):
        actor_output = self.actor(state)
        mean, log_std = torch.chunk(actor_output, 2, dim=-1)
        mean = torch.tanh(mean)
        std = torch.clamp(log_std.exp(), min=0.2, max=1.0)
        return torch.distributions.Normal(mean, std)
    
    def get_value(self, state):
        return self.critic(state)

def normalize_state(state):
    pos = state[0] / 1.2
    vel = state[1] / 0.07
    return np.array([pos, vel], dtype=np.float32)

def train():
    output_dir = "mountain_car_results"
    os.makedirs(output_dir, exist_ok=True)
    
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    model = ActorCritic()
    metrics = TrainingMetrics()
    recorder = AnimationRecorder(output_dir)
    
    actor_optimizer = optim.Adam(model.actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)
    
    max_episodes = 1000
    max_steps = 999
    gamma = 0.99
    gae_lambda = 0.95
    
    exploration_noise = 0.5
    noise_decay = 0.995
    min_noise = 0.1
    
    for episode in range(max_episodes):
        # Start recording at episode 10
        if episode == 10:
            recorder.start_recording()
            
        state, _ = env.reset()
        state = normalize_state(state)
        episode_reward = 0
        highest_position = -1.2
        episode_critic_loss = 0
        n_updates = 0
        
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        
        episode_entropy = 0
        
        # Collect trajectory
        for step in range(max_steps):
            # Record frame if we're recording
            recorder.add_frame(env)
            
            state_tensor = torch.FloatTensor(state)
            
            with torch.no_grad():
                dist = model.get_action_dist(state_tensor)
                value = model.get_value(state_tensor)
                
                action = dist.sample()
                if episode < 100:
                    action += torch.randn_like(action) * exploration_noise
                action = torch.clamp(action, -1, 1)
                log_prob = dist.log_prob(action).sum()
            
            action_np = action.numpy()
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            
            done = terminated or truncated
            highest_position = max(highest_position, next_state[0])
            
            # Reward shaping
            shaped_reward = reward
            if next_state[0] > state[0]:
                shaped_reward += (next_state[0] - state[0]) * 10
            
            if abs(next_state[1]) > abs(state[1]):
                shaped_reward += (abs(next_state[1]) - abs(state[1])) * 5
            
            if abs(next_state[0]) < 0.1:
                shaped_reward -= 0.1
            
            if done and next_state[0] >= 0.45:
                shaped_reward = 100.0
                # if not recorder.recording:  # Start recording successful episode
                #     recorder.start_recording()
                #     # Replay the episode for recording
                #     state, _ = env.reset()
                #     continue
            
            states.append(state_tensor)
            actions.append(action)
            rewards.append(shaped_reward)
            values.append(value.item())
            log_probs.append(log_prob)
            
            episode_reward += reward
            state = normalize_state(next_state)
            
            if done:
                break
        
        # Save episode 10 recording
        if episode == 10:
            recorder.save_gif(episode)
        
        # Prepare for PPO updates
        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs)
        
        # Calculate advantages using GAE
        with torch.no_grad():
            next_value = 0 if done else model.get_value(torch.FloatTensor(state)).item()
            returns = []
            advantages = []
            gae = 0
            
            for t in reversed(range(len(rewards))):
                next_val = next_value if t == len(rewards)-1 else values[t+1]
                delta = rewards[t] + gamma * next_val - values[t]
                gae = delta + gamma * gae_lambda * gae
                
                returns.insert(0, gae + values[t])
                advantages.insert(0, gae)
            
            returns = torch.tensor(returns)
            advantages = torch.tensor(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO updates
        for _ in range(4):
            # Actor update
            dist = model.get_action_dist(states)
            new_log_probs = dist.log_prob(actions).sum(-1)
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            
            # Calculate approximate KL divergence for tracking
            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            
            # PPO clipped objective
            actor_loss1 = -ratio * advantages
            actor_loss2 = -torch.clamp(ratio, 0.8, 1.2) * advantages
            actor_loss = torch.max(actor_loss1, actor_loss2).mean()
            entropy_loss = -0.02 * dist.entropy().mean()
            
            total_actor_loss = actor_loss + entropy_loss
            
            actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)
            actor_optimizer.step()
            
            # Critic update
            values = model.get_value(states).squeeze()
            critic_loss = (returns - values).pow(2).mean()
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 0.5)
            critic_optimizer.step()
            
            episode_critic_loss += critic_loss.item()
            episode_entropy += dist.entropy().mean().item()
            n_updates += 1
        
        exploration_noise = max(min_noise, exploration_noise * noise_decay)
        
        # Update metrics
        metrics.update(
            reward=episode_reward,
            length=step+1,
            critic_loss=episode_critic_loss/n_updates if n_updates > 0 else 0,
            actor_model=model  # Pass the whole model to extract weights
        )
        
        # Plot every 10 episodes
        if episode % 10 == 0:
            metrics.plot(output_dir, episode)
            print(f"Episode {episode}, Steps: {step+1}, Reward: {episode_reward:.1f}, "
                  f"Max Position: {highest_position:.2f}, Mean Loss: {episode_critic_loss/n_updates:.3f}")
        
        # Save successful episode recording and check for solving
        if episode_reward >= 96.5:
            print(f"Solved in {episode} episodes!")
            # if recorder.recording:
            #     recorder.save_gif(episode)
            metrics.plot(output_dir, episode)
            break
    
    env.close()
    return model

if __name__ == "__main__":
    train()

