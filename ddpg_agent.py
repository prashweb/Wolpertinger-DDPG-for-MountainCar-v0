import numpy as np
import os
import random
from collections import deque
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════
# Network Utilities
# ══════════════════════════════════════════════════════════════════════
def fanin_uniform(tensor):
    fan_in = tensor.size(0)
    bound = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -bound, bound)

# ══════════════════════════════════════════════════════════════════════
# Actor Network
# ══════════════════════════════════════════════════════════════════════
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dims=256, fc2_dims=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, action_dim) # Outputs action

        self.ln1 = nn.LayerNorm(fc1_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        self.reset_parameters()

    def reset_parameters(self):
        fanin_uniform(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        fanin_uniform(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3) # Small weights for output layer
        nn.init.zeros_(self.fc3.bias)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        action = T.tanh(self.fc3(x)) # Squash actions to [-1, 1]
        return action

# ══════════════════════════════════════════════════════════════════════
# Critic Network
# ══════════════════════════════════════════════════════════════════════
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dims=256, fc2_dims=256, fc3_dims=256):
        super(Critic, self).__init__()
        # Processing state
        self.state_fc = nn.Linear(state_dim, fc1_dims)
        self.state_ln = nn.LayerNorm(fc1_dims)

        # Processing action
        self.action_fc = nn.Linear(action_dim, fc2_dims)
        self.action_ln = nn.LayerNorm(fc2_dims)

        # Combined processing
        self.fc_merge = nn.Linear(fc1_dims + fc2_dims, fc3_dims) # Concatenate processed state and action
        self.merge_ln = nn.LayerNorm(fc3_dims)
        self.fc_out = nn.Linear(fc3_dims, 1) # Output Q-value
        self.reset_parameters()

    def reset_parameters(self):
        fanin_uniform(self.state_fc.weight)
        nn.init.zeros_(self.state_fc.bias)
        fanin_uniform(self.action_fc.weight)
        nn.init.zeros_(self.action_fc.bias)
        fanin_uniform(self.fc_merge.weight)
        nn.init.zeros_(self.fc_merge.bias)
        nn.init.uniform_(self.fc_out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, state, action):
        s_out = F.relu(self.state_ln(self.state_fc(state)))
        a_out = F.relu(self.action_fc(action)) # No LayerNorm on action here, often better

        x = T.cat([s_out, a_out], dim=1)
        x = F.relu(self.merge_ln(self.fc_merge(x)))
        q_value = self.fc_out(x)
        return q_value

# Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha=0.6,
                 beta_start=0.4, beta_end=1.0,
                 total_anneal_steps=1_000_000):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.max_size = max_size
        self.alpha = alpha
        self.beta  = beta_start
        self.b0, self.b1 = beta_start, beta_end
        self.total_steps = total_anneal_steps
        self.step_cnt = 0
        self.epsilon = 1e-6
        
    """def add(self, transition, priority=None):
        if priority is None:
            priority = max(self.priorities) if self.buffer else 1.0
        self.buffer.append(transition)
        self.priorities.append(priority)"""
    """def add(self, tr, td_err=None):
        p = (abs(td_err) + self.epsilon) if td_err is not None else 1.0
        self.buffer.append(tr); self.priorities.append(p)"""
    def add(self, transition, p=None):
        p = max(self.priorities) if (p is None and self.priorities) else (p or 1.0)
        self.buffer.append(transition)
        self.priorities.append(float(p))

        
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], []
    
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return samples, indices, weights
        
    """def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + self.epsilon"""
    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            pri = float(abs(err)) + self.epsilon
            self.priorities[idx] = pri ** self.alpha

    def anneal_beta(self, n=1):
        self.step_cnt += n
        f = min(1.0, self.step_cnt / self.total_steps)
        self.beta = self.b0 + f * (self.b1 - self.b0)

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim,
                 fc1_dims=256, fc2_dims=256, fc3_dims=256, # Network layer sizes
                 buffer_size=int(1e6), gamma=0.99, tau=0.005,
                 policy_delay=2, actor_lr=1e-4, critic_lr=1e-3, # Adjusted critic LR
                 warmup_steps=1000, batch_size=64, # Common batch size
                 chkpt_dir='chkpt_td3'):
        
        os.makedirs(chkpt_dir, exist_ok=True)
        self.chkpt_dir = chkpt_dir
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # Actor Networks
        self.actor = Actor(state_dim, action_dim, fc1_dims, fc2_dims).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, fc1_dims, fc2_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic Networks
        self.critic = Critic(state_dim, action_dim, fc1_dims, fc2_dims, fc3_dims).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, fc1_dims, fc2_dims, fc3_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
 
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        self.noise = GaussianNoise(action_dim)

        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size

        self.total_env_steps = 0 # For warmup check
        self.learn_step = 0

        self.actor_losses = []
        self.critic_losses = []
        
    def select_action(self, state, add_noise=True):
        state_t = T.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state_t).squeeze(0).detach().cpu().numpy()
        if add_noise:action = np.clip(action + self.noise.sample(), -1, 1)
        return action
    
    def store_transition(self, state, action, reward, next_state, terminated, truncated):
        self.total_env_steps += 1            
        # Calculate initial priority using TD error
        with T.no_grad():
            state_t      = T.as_tensor(state,      device=self.device, dtype=T.float32).unsqueeze(0)
            action_t     = T.as_tensor(action,     device=self.device, dtype=T.float32).unsqueeze(0)
            next_state_t = T.as_tensor(next_state, device=self.device, dtype=T.float32).unsqueeze(0)
            reward_t     = T.as_tensor([reward],   device=self.device, dtype=T.float32)     # NEW
            done_t       = T.as_tensor([terminated],     device=self.device, dtype=T.float32)     # NEW (float!)

            current_q      = self.critic(state_t, action_t)
            next_action    = self.actor_target(next_state_t)
            target_q_next  = self.critic_target(next_state_t, next_action)

            # (1 - done_t) now works because done_t is float32 (0. or 1.)
            target_q = reward_t + self.gamma * target_q_next * (1.0 - done_t)
            td_error = (current_q - target_q).abs().item()
            
        # Store transition with priority
        self.replay_buffer.add((state, action, reward, next_state, terminated, truncated), td_error)
        self.replay_buffer.anneal_beta()
    
    def train(self):
        if len(self.replay_buffer.buffer) < max(self.batch_size, self.warmup_steps):
            return
        
        # Sample from replay buffer
        batch, indices, weights = self.replay_buffer.sample(self.batch_size)

        states, actions, rewards, next_states, terminations, truncations = map(np.array, zip(*batch))

        states  = T.as_tensor(states,  device=self.device, dtype=T.float32)
        actions = T.as_tensor(actions, device=self.device, dtype=T.float32)
        rewards = T.as_tensor(rewards, device=self.device, dtype=T.float32).unsqueeze(1)
        next_states = T.as_tensor(next_states, device=self.device, dtype=T.float32)
        dones   = T.as_tensor(terminations,   device=self.device, dtype=T.float32).unsqueeze(1)
        is_weights = T.as_tensor(weights, device=self.device, dtype=T.float32).unsqueeze(1)
        
        # Update Critic
        with T.no_grad():
            next_actions_target = (self.actor_target(next_states) +
                    (0.2*T.randn_like(actions)).clamp(-0.5, 0.5)).clamp(-1, 1)
            target_Q = self.critic_target(next_states, next_actions_target)
            y_target = rewards + (self.gamma * target_Q) * (1 - dones)
        
        current_Q = self.critic(states, actions)
        td_errors = (current_Q - y_target).abs().detach().cpu().numpy().flatten()
        loss_c = (is_weights * F.mse_loss(current_Q, y_target, reduction='none')).mean()
        
        self.critic_optimizer.zero_grad()
        loss_c.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # Add gradient clipping
        self.critic_optimizer.step()
        
        """# Actor Update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # Clip actor grads
        self.actor_optimizer.step()
        self.actor_losses.append(actor_loss.item())"""
        # ─────────────────────────  actor update  ───────────────────────
        if self.learn_step % self.policy_delay == 0:                     
            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            T.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
        # soft-update targets
            for t, s in zip(self.actor_target.parameters(), self.actor.parameters()):
                t.data.mul_(1 - self.tau).add_(self.tau * s.data)
        else:
            actor_loss = T.tensor(0.)
        
        # Soft update target networks
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.mul_(1 - self.tau).add_(self.tau * s.data)

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(loss_c.item())
                    
        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)
        self.learn_step += 1
    
    def save(self, filename_prefix="ddpg_agent"):
        T.save(self.actor.state_dict(), os.path.join(self.chkpt_dir, f"{filename_prefix}_actor.pth"))
        T.save(self.actor_target.state_dict(), os.path.join(self.chkpt_dir, f"{filename_prefix}_actor_target.pth"))
        T.save(self.critic.state_dict(), os.path.join(self.chkpt_dir, f"{filename_prefix}_critic.pth"))
        T.save(self.critic_target.state_dict(), os.path.join(self.chkpt_dir, f"{filename_prefix}_critic_target.pth"))
        # Also save optimizers if you want to resume training exactly
        T.save(self.actor_optimizer.state_dict(), os.path.join(self.chkpt_dir, f"{filename_prefix}_actor_optimizer.pth"))
        T.save(self.critic_optimizer.state_dict(), os.path.join(self.chkpt_dir, f"{filename_prefix}_critic_optimizer.pth"))
        print(f"... saving models to {self.chkpt_dir} ...")

    def load(self, filename_prefix="ddpg_agent", evaluate=False):
        print(f"... loading models from {self.chkpt_dir} ...")
        self.actor.load_state_dict(T.load(os.path.join(self.chkpt_dir, f"{filename_prefix}_actor.pth"), map_location=self.device))
        self.actor_target.load_state_dict(T.load(os.path.join(self.chkpt_dir, f"{filename_prefix}_actor_target.pth"), map_location=self.device))
        self.critic.load_state_dict(T.load(os.path.join(self.chkpt_dir, f"{filename_prefix}_critic.pth"), map_location=self.device))
        self.critic_target.load_state_dict(T.load(os.path.join(self.chkpt_dir, f"{filename_prefix}_critic_target.pth"), map_location=self.device))
        if not evaluate: # Only load optimizers if continuing training
            self.actor_optimizer.load_state_dict(T.load(os.path.join(self.chkpt_dir, f"{filename_prefix}_actor_optimizer.pth"), map_location=self.device))
            self.critic_optimizer.load_state_dict(T.load(os.path.join(self.chkpt_dir, f"{filename_prefix}_critic_optimizer.pth"), map_location=self.device))
        if evaluate:
            self.actor.eval()
            self.critic.eval()

    def get_loss_history(self):
        return self.actor_losses, self.critic_losses

    @staticmethod
    def moving_average(x, k=100):
        x = np.asarray(x, dtype=np.float32)
        if x.size < k: return x
        c = np.cumsum(np.insert(x, 0, 0))
        return (c[k:] - c[:-k]) / k
    
# Ornstein-Uhlenbeck noise for exploration
""" class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state """

#Gaussian noise for exploration
class GaussianNoise:
    def __init__(self, action_dim, initial_scale=0.6, min_scale=0.05, decay_rate=0.999995):
        self.action_dim = action_dim
        self.scale = initial_scale
        self.min_scale = min_scale
        self.decay_rate = decay_rate

    def sample(self):
        noise = self.scale * np.random.randn(self.action_dim)
        self.scale = max(self.min_scale, self.scale * self.decay_rate)
        return noise
    
    def reset(self):
        pass  # No internal state needed
