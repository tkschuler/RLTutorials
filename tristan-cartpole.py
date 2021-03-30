# objective is to get the cart to the flag.
# for now, let's just move randomly:

import gym
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style

env = gym.make("CartPole-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10000
SHOW_EVERY = 200
UPDATE_EVERY = STATS_EVERY = 100


#print(env.reward_range)

# For stats
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}


# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


# Number of stepsizes for discrete oberservation space
# Don't all have to be the same, but they are in this case.
n = 20

#discrete observation space
discrete_os = [ np.linspace(-4.8, 4.8, n),
                np.linspace(-4, 4, n),
                np.linspace(-.418, .418, n),
                np.linspace(-20, 20, n)
              ]


# How many states can be observed
os_size = len(env.observation_space.high)

# Initialize Q-Table with random numbers
# The size will be multi-dimensional, driven by observation space size
# (n, n, n,...., a)
q_table = np.random.uniform(low=-2, high=0, size=([n] * os_size + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = []
    for i in range(os_size):
        discrete_state.append(np.digitize(state[i], discrete_os[i]) - 1)  # Need -1, otherwise array starts indexing at 1 instead of 0
    return tuple(discrete_state)

previousCnt = []  # array of all scores over runs
metrics = {'ep': [], 'avg': [], 'min': [], 'max': []}  # metrics recorded for graph


for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False

    cnt = 0
    episode_reward = 0

    '''
    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    '''


    while not done:

        if episode % SHOW_EVERY == 0:
            env.render()  # if running RL comment this out

        cnt += 1

        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)


        new_state, reward, done, info = env.step(action)

        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)


        if episode % SHOW_EVERY == 0:
            env.render()
        #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # If simulation did not end yet after last step - update Q table


        #print(new_discrete_state)
        #print(q_table[new_discrete_state])
        # Maximum possible Q value in next step (for new state)
        max_future_q = np.max(q_table[new_discrete_state])

        # Current Q value (for current state and performed action)
        current_q = q_table[discrete_state + (action,)]

        # pole fell over / went out of bounds, negative reward
        #'''
        if done and cnt < 200:
            reward = -375
       # '''

        # And here's our equation for a new Q value for current state and action
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        #print(new_q)
        # Update Q table with new Q value
        q_table[discrete_state + (action,)] = new_q


        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        '''
        elif new_state[0] >= env.goal_position:
            #q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0
            #print(episode, print(info))
        '''

        discrete_state = new_discrete_state

    previousCnt.append(cnt)

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value


    if episode % UPDATE_EVERY == 0:
        latestRuns = previousCnt[-UPDATE_EVERY:]
        averageCnt = sum(latestRuns) / len(latestRuns)
        metrics['ep'].append(episode)
        metrics['avg'].append(averageCnt)
        metrics['min'].append(min(latestRuns))
        metrics['max'].append(max(latestRuns))
        print("Run:", episode, "Average:", averageCnt, "Min:", min(latestRuns), "Max:", max(latestRuns))

    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')
        #np.save(f"qtables/{episode}-qtable.npy", q_table)



env.close()

# Plot graph
plt.figure(1)
plt.plot(metrics['ep'], metrics['avg'], label="average rewards")
plt.plot(metrics['ep'], metrics['min'], label="min rewards")
plt.plot(metrics['ep'], metrics['max'], label="max rewards")
plt.legend(loc=4)

plt.figure(2)
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.legend(loc=4)
plt.show()

