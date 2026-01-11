import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import trange

class RobustDistEnv:
    # State initialization, sampling goals, and selecting a training distribution
    def __init__(self, size=5, n_goals=2, max_steps=100, distributions=None, seed=None):
        self.size = size
        self.n_goals = n_goals
        self.max_steps = max_steps

        if distributions is None:
            distributions = [
                {(x, y): 1/(size*size) for x in range(size) for y in range(size)},
                {(0,0): 0.5, (size-1,size-1): 0.5},
                {(size//2, size//2): 1.0},
            ]
        self.distributions = distributions

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.reset()

    # Reset agent position, goals, and select a new distribution
    def reset(self):
        self.steps = 0
        self.current_dist = random.choice(self.distributions)
        self.agent_pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))

        self.goals = set()
        while len(self.goals) < self.n_goals:
            g = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if g != self.agent_pos:
                self.goals.add(g)

        return self.get_state()

    # Return hashable state representation
    def get_state(self):
        return (tuple(self.agent_pos), frozenset(self.goals))

    # Identify out-of-distribution states based on current distribution probability
    def is_ood(self, pos):
        return self.current_dist.get(tuple(pos), 0.0) <= 1e-10

    # Q-learning step with ε-greedy, reward shaping for OOD, and goal completion
    def step(self, action):
        self.steps += 1
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = moves[action]
        new_x = int(np.clip(self.agent_pos[0] + dx, 0, self.size - 1))
        new_y = int(np.clip(self.agent_pos[1] + dy, 0, self.size - 1))
        self.agent_pos = (new_x, new_y)

        reward = -0.1
        ood = self.is_ood(self.agent_pos)

        if ood:
            reward -= 0.8

        success = False
        if self.agent_pos in self.goals:
            reward += 12.0
            self.goals.remove(self.agent_pos)
            success = True

        done = (len(self.goals) == 0) or (self.steps >= self.max_steps)

        return self.get_state(), reward, ood, done, success


# Q-learning with ε-greedy, bootstrap updates, OOD penalty
def train_robust_q(n_episodes=12000, alpha=0.08, gamma=0.98,
                  epsilon_start=1.0, epsilon_end=0.015, epsilon_decay_frac=0.75,
                  seed=42):

    env = RobustDistEnv(size=5, n_goals=2, max_steps=100, seed=seed)
    Q = defaultdict(float)
    rewards, ood_steps, success_rate = [], [], []

    for ep in trange(n_episodes, desc="Robust Q"):
        state = env.reset()
        episode_reward = 0.0
        episode_ood = 0
        finished_all_goals = False

        epsilon = max(epsilon_end,
                     epsilon_start * (1 - ep / (n_episodes * epsilon_decay_frac)))

        while True:
            if random.random() < epsilon:
                action = random.randrange(4)
            else:
                q_vals = [Q[(state, a)] for a in range(4)]
                max_q = max(q_vals)
                candidates = [a for a, q in enumerate(q_vals) if q >= max_q - 1e-9]
                action = random.choice(candidates)

            next_state, r, is_ood, done, goal_reached = env.step(action)

            episode_reward += r
            episode_ood += is_ood

            if done and len(next_state[1]) == 0:
                target = r
                finished_all_goals = True
            else:
                next_best = max(Q[(next_state, a)] for a in range(4))
                target = r + gamma * next_best

            Q[(state, action)] += alpha * (target - Q[(state, action)])
            state = next_state

            if done:
                break

        rewards.append(episode_reward)
        ood_steps.append(episode_ood)
        success_rate.append(finished_all_goals)

    return rewards, ood_steps, success_rate, Q, env


if __name__ == "__main__":
    rewards, oods, successes, Q, env = train_robust_q(n_episodes=15000, seed=42)

    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    window = 150

    axs[0].plot(np.convolve(rewards,  np.ones(window)/window, mode='valid'), 'royalblue', lw=1.1)
    axs[0].set_title("Average Episode Reward")
    axs[0].set_ylabel("Reward")

    axs[1].plot(np.convolve(oods, np.ones(window)/window, mode='valid'), 'darkorange', lw=1.1)
    axs[1].set_title("Average OOD steps per episode")
    axs[1].set_ylabel("OOD count")

    axs[2].plot(np.convolve(successes, np.ones(window)/window, mode='valid'), 'forestgreen', lw=1.2)
    axs[2].set_title("Success rate (all goals reached)")
    axs[2].set_ylabel("Success rate")
    axs[2].set_xlabel("Episode")

    for ax in axs:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Final success rate (last {window} episodes): {np.mean(successes[-window:]):.3%}")
