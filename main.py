# Ref:
# https://github.com/Kaixhin/Rainbow/blob/master/main.py
# https://github.com/Curt-Park/rainbow-is-all-you-need
# https://github.com/Huixxi/TensorFlow2.0-for-Deep-Reinforcement-Learning

from utils.Agent import Agent
import gym
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as kb
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
kb.clear_session()


fig, axs = plt.subplots(1, 2, figsize=(20, 7))


def plot(scores, ma, losses):
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


env = gym.make("LunarLander-v2")

n_frame = 100_000
agent = Agent([56, 56, 3], env.action_space.n, batch_size=64, v_max=300.0,
              v_min=-400.0)
history = list()
ma = list()
losses = list()

env.reset()
current_state = tf.image.resize(env.render(mode="rgb_array"), (56, 56)) / 255.0
done = False
episode_reward = 0

for frame in range(n_frame):
    action = agent.get_action(current_state)
    _, reward, done, _ = env.step(action)
    future_state = tf.image.resize(env.render(mode="rgb_array"), (56, 56)) / 255.0
    loss = agent.step(current_state, reward, future_state, done, action)

    current_state = future_state
    episode_reward += reward

    losses.append(loss)

    if done:
        agent.decay_epsilon()
        env.reset()
        current_state = tf.image.resize(env.render(mode="rgb_array"), (56, 56)) / 255.0

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


plt.show()

for _ in range(10):
    done = False
    env.reset()
    state = tf.image.resize(env.render(mode="rgb_array"), (56, 56)) / 255.0
    rwd = 0
    while not done:
        action = agent.get_action(current_state, testing=True)
        _, reward, done, _ = env.step(action)
        state = tf.image.resize(env.render(mode="rgb_array"), (56, 56)) / 255.0
        rwd += reward
    print(rwd)

env.close()


