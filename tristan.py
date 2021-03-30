import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95

EPISODES = 10000
SHOW_EVERY = 1000

# Determine the ranges for the observation space
print(env.observation_space.high)
print(env.observation_space.low)


# Number of stepsizes for discrete oberservation space
# Don't all have to be the same, but they are in this case.
n = 20

#discrete observation space
discrete_os = [ np.linspace(-1.2, .06, n),
                np.linspace(-.07,.07, n)
              ]


# How many states can be observed
os_size = len(env.observation_space.high)

# Initialize Q-Table with random numbers
# The size will be multi-dimensional, driven by observation space size
# (n, n, n,...., a)
q_table = np.random.uniform(low=-2, high=0, size=([n] * os_size + [env.action_space.n]))


def get_discrete_state(state):
    """
    Determine index values of continuous state in the discrete space.
    """
    stateIndex = []
    for i in range(os_size):
        stateIndex.append(np.digitize(state[i], discrete_os[i]) - 1)  # Need -1, otherwise array starts indexing at 1 instead of 0
    return stateIndex



done = False
discrete_state = get_discrete_state(env.reset())


for episode in range(EPISODES):
    done = False
    discrete_state = get_discrete_state(env.reset())

    if episode % SHOW_EVERY == 0:
        print(episode)


    while not done:
        #action = 0  # always go left!

        q_index = tuple(discrete_state)
        action = np.argmax(q_table[q_index]) # best current action, from highest q value

        #print(tuple(discrete_state + [action]))
        new_state, reward, done, info = env.step(action)  #do the action

        new_discrete_state = tuple(get_discrete_state(new_state))
        new_q_index = tuple(new_discrete_state)

        if episode % SHOW_EVERY == 0:
            env.render()

        #env.render()

        if not done:

            max_future_q = np.max(q_table[new_q_index]) #Best future q value

            current_q = q_table[tuple(discrete_state + [action])]  # current q value for observed state and best action

            # q learning equation:
            q_new = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            #print("q_new", q_new)
            q_table[tuple(discrete_state + [action])] = q_new

        elif new_state[0] >= env.goal_position:
            #q_table[discrete_state + (action,)] = reward
            q_table[tuple(discrete_state) + (action,)] = 0
            print(episode)

        discrete_state = np.asarray(new_discrete_state)

    #print(reward)