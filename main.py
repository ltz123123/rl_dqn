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
    axs[0].plot(np.arange(len(scores)), scores, linewidth=0.2, color="b")
    axs[0].plot(np.arange(len(ma)) + 99, ma, color="r")
    axs[0].set_ylabel("Episode reward")
    axs[0].set_xlabel("Episode")
    axs[0].grid(1)

    axs[1].plot(np.arange(len(losses)), losses, linewidth=0.1, color="b")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Frame")
    axs[1].grid(1)
    fig.savefig("fig.jpg")


def eval(test_env, test_agent, image_size, n_eval_episode=100):
    test_score_list = []
    for _ in range(n_eval_episode):
        done = False
        state = test_env.reset()
        # state = tf.image.resize(env.render(mode="rgb_array"), image_size[:-1]) / 255.0
        rwd = 0
        while not done:
            action = test_agent.get_action(state, training=False)
            state, reward, done, _ = test_env.step(action)
            # state = tf.image.resize(env.render(mode="rgb_array"), image_size[:-1]) / 255.0
            rwd += reward

        test_score_list.append(rwd)

    env.close()

    avg = np.average(test_score_list)
    print(f"Test score: {avg}")

    return avg


# env_id = "LunarLander-v2"
env_id = "CartPole-v1"
n_frame = 100_000
img_size = [116, 116, 3]

env = gym.make(env_id)
agent = Agent(
    [4], env.action_space.n, update_target_model_every=1000, batch_size=128, v_max=500.0, v_min=0.0, n_step=3,
    n_atom=51, max_memory_len=2 ** 14, is_noisy=False, is_img_input=False
)
episode_reward_list = list()
ma = list()
losses = list()

current_state = env.reset()
# current_state = tf.image.resize(env.render(mode="rgb_array"), img_size[:-1]) / 255.0
done = False
episode_reward = 0
episode_frame_taken = 0

for frame in range(n_frame):
    action = agent.get_action(current_state)
    future_state, reward, done, _ = env.step(action)
    # future_state = tf.image.resize(env.render(mode="rgb_array"), img_size[:-1]) / 255.0
    loss = agent.step(current_state, reward, future_state, done, action, frame)

    current_state = future_state
    episode_reward += reward
    episode_frame_taken += 1
    losses.append(loss)

    if done:
        episode_reward = round(episode_reward, 4)
        episode_reward_list.append(episode_reward)
        print(f"Episode: {len(episode_reward_list): >4}, frame taken:{episode_frame_taken: >4}, reward: {episode_reward}")
        episode_frame_taken = 0
        episode_reward = 0

        if len(episode_reward_list) >= 100:
            ma.append(np.mean(episode_reward_list[-100:]))

        if len(episode_reward_list) % 50 == 0 and frame > 0:
            avg_score = eval(env, agent, img_size, n_eval_episode=10)
            if avg_score >= 500:
                break

        agent.decay_epsilon()
        done = False
        current_state = env.reset()
        # current_state = tf.image.resize(env.render(mode="rgb_array"), img_size[:-1]) / 255.0

    if frame % 500 == 0:
        plot(episode_reward_list, ma, losses)


plot(episode_reward_list, ma, losses)
plt.show()
env.close()


