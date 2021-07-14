from utils.Agent import Agent
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import backend as kb

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
kb.clear_session()


def plot(scores, ma, losses):
    fig, axs = plt.subplots(1, 2, figsize=(10, 7))
    axs[0].plot(np.arange(len(scores)), scores, linewidth=0.5)
    axs[0].plot(np.arange(len(ma)) + 99, ma)
    axs[0].set_ylabel("Episode reward")
    axs[0].set_xlabel("Episode")
    axs[0].grid(1)

    axs[1].plot(np.arange(len(losses)), losses, linewidth=0.5)
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Frame")
    axs[1].grid(1)
    fig.savefig("fig.jpg")


env = gym.make("CartPole-v1")

n_frame = 100_000
agent = Agent(len(env.observation_space.high), env.action_space.n, batch_size=64, v_max=500.0,
              v_min=0.0)
history = list()
ma = list()
losses = list()

current_state = env.reset()
done = False
episode_reward = 0

for frame in range(n_frame):
    action = agent.get_action(current_state)
    future_state, reward, done, _ = env.step(action)
    loss = agent.step(current_state, reward, future_state, done, action)

    current_state = future_state
    episode_reward += reward

    losses.append(loss)

    if done:
        agent.decay_epsilon()
        current_state = env.reset()

        episode_reward = round(episode_reward, 4)
        history.append(episode_reward)
        print(f"{frame}, {len(history)}, {episode_reward}")

        episode_reward = 0

        if len(history) >= 100:
            ma.append(np.mean(history[-100:]))
            if ma[-1] >= 200:
                break

    if frame % 500 == 0:
        plot(history, ma, losses)

env.close()


