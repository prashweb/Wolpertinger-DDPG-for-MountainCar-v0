"""
Wolpertinger-DDPG on Gymnasium’s MountainCar-v0
------------------------------------------------
The Actor in the supplied DDPGAgent produces a continuous value in the
range [-1, 1].  We wrap the agent with a Wolpertinger policy that

1. takes the proto–action aᶜ  coming from the actor,
2. finds the k nearest discrete actions in an embedding space
   (here k = 3 and the embedding is simply
        [[-1.],            # env action 0  (push left)
         [ 0.],            # env action 1  (no push)
         [ 1.]]),          # env action 2  (push right)
3. evaluates those candidates with the critic and
4. executes / stores the best one
"""

import gym
import numpy as np
import torch as T
from collections import deque
from pathlib import Path
import time
from ddpg_agent import DDPGAgent


# ════════════════════════════════════════════════════════════════════════
#               Wolpertinger policy wrapper around DDPGAgent
# ════════════════════════════════════════════════════════════════════════
class WolpertingerPolicy:
    """
    Maps the continuous proto-action produced by the actor to a discrete
    Mountain-Car action via k-NN + critic re-ranking (Wolpertinger).
    """
    def __init__(self, ddpg_agent, env_action_space, k_ratio: float = 1.0):
        """
        k_ratio · |A| nearest neighbours are considered (≥ 1).
        For MountainCar-v0 |A| = 3, so k = 3 gives the whole set.
        """
        self.agent = ddpg_agent
        self.discrete_actions = np.arange(env_action_space.n)      # [0, 1, 2]
        # Simple 1-D embedding that matches the actor’s output dimension
        self.embeddings = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)
        self.k = max(1, int(k_ratio * env_action_space.n))

    # ------------------------------------------------------------------
    def act(self, state, add_noise=True):
        """
        Returns
            env_action  … integer sent to env.step()
            cont_embed  … continuous 1-D vector fed to the critic / replay-buffer
        """
        # 1) proto-action from actor
        proto_action = self.agent.select_action(state, add_noise=add_noise)  # shape (1,)
        # 2) k-NN in embedding space
        dists = np.linalg.norm(self.embeddings - proto_action, axis=1)
        nn_indices = dists.argsort()[:self.k]                                # indices into embeddings
        # 3) critic re-ranking
        q_vals = [self.agent.evaluate_q(state, self.embeddings[i])
                  for i in nn_indices]
        best_idx = nn_indices[int(np.argmax(q_vals))]
        env_action = int(self.discrete_actions[best_idx])
        cont_embed = self.embeddings[best_idx]                                # shape (1,)
        return env_action, cont_embed

    # ------------------------------------------------------------------
    def eval_policy(self, env, episodes=5):
        """
        Runs episodes without exploration noise.  Returns average return.
        """
        scores = []
        for _ in range(episodes):
            s, _ = env.reset()
            done, trunc = False, False
            ep_ret = 0.0
            while not (done or trunc):
                a, _ = self.act(s, add_noise=False)
                s, r, done, trunc, _ = env.step(a)
                ep_ret += r
            scores.append(ep_ret)
        return np.mean(scores)


# ════════════════════════════════════════════════════════════════════════
#                         Training loop
# ════════════════════════════════════════════════════════════════════════
def train_mountaincar(
        episodes: int           = 600,
        max_steps: int          = 200,
        eval_every: int         = 25,
        render:   bool          = False,
        seed:     int           = 1):

    env = gym.make('MountainCar-v0', render_mode='human')
    env.reset(seed=seed)
    np.random.seed(seed); T.manual_seed(seed)

    state_dim  = env.observation_space.shape[0]      # (=2)
    action_dim = 1                                   # continuous embedding dimension

    agent   = DDPGAgent(state_dim, action_dim,
                        batch_size=128,
                        warmup_steps=5_000,
                        chkpt_dir='chkpt_wolpertinger')
    policy  = WolpertingerPolicy(agent, env.action_space, k_ratio=1.0)

    ep_returns     = deque(maxlen=eval_every)
    eval_history   = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = truncated = False
        ep_return = 0.0

        for t in range(max_steps):
            if render: env.render()

            # ───────── Choose & execute action ─────────
            env_act, act_embed = policy.act(state, add_noise=True)
            next_state, reward, done, truncated, _ = env.step(env_act)

            # MountainCar gives reward = -1 every step until goal is reached
            # Convert the  termination signal to float for the agent
            agent.store_transition(state,
                                   act_embed,  # continuous 1-D action fed to critic
                                   reward,
                                   next_state,
                                   terminated=done,
                                   truncated=truncated)

            agent.train()

            state = next_state
            ep_return += reward
            if done or truncated: break

        ep_returns.append(ep_return)

        # Periodic evaluation (without noise)
        if ep % eval_every == 0:
            avg_eval_ret = policy.eval_policy(env, episodes=5)
            eval_history.append((ep, avg_eval_ret))
            print(f"[{time.strftime('%H:%M:%S')}] Episode {ep:4d} "
                  f"train-R = {np.mean(ep_returns):6.1f} | "
                  f"eval-R = {avg_eval_ret:6.1f} | "
                  f"buffer = {len(agent.replay_buffer.buffer):6d}")

    env.close()
    return eval_history


# ════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    train_mountaincar(episodes=600, render=True)