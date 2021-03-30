import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')


def q_learning(episode):

    score_list = []
    alpha = .8
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(episode):
        done = False
        score, reward = 0,0
        state = env.reset()

        while not done:

            env.render()
            action = np.argmax(Q[state])
            state2, reward, done, info = env.step(action)

            target = reward + np.max(Q[state2]) - Q[state, action]
            Q[state, action] += alpha * target
            score += reward
            state = state2

        score_list.append(score)
        print("episode {}, total reward = {}".format(i, score))

    return score_list


def random_policy(episode, step):

    for i_episode in range(episode):
        env.reset()
        for t in range(step):
            env.render()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        print("Starting next episode")


if __name__ == '__main__':

    ep = 400
    score_list = q_learning(ep)
    plt.plot([i+1 for i in range(0, ep, 4)], score_list[::4])
    plt.show()