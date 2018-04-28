import numpy as np
import gym
from pendulum_task import PendulumTask
from agents.agent import DDPG

task = PendulumTask()
agent = DDPG(task)
env = gym.make("Pendulum-v0")

done = False
n_episodes = 400
rewards = np.zeros(n_episodes)
for i in range(n_episodes):
    cur_state = env.reset()
    agent.reset_episode(cur_state)
    while True:
        env.render()
        random_action = env.action_space.sample()
        action = agent.act(cur_state)

        new_state, reward, done, _ = env.step(action)
        rewards[i] += reward

        #train step
        agent.step(action, reward, new_state, done)

        if done:
            print("\rEpisode = {:4d}, total_reward = {:7.3f}".format(i, rewards[i]))
            break
        else:
            cur_state = new_state