import gym

env = gym.make('CartPole-v0')

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

def policy(t):
    action = 0
    if t%2 == 1:  # if the time step is odd
        action = 1
    return action


nb_episodes = 20
nb_timesteps = 100

for episode in range(nb_episodes):  # iterate over the episodes
    state = env.reset()  # initialise the environment
    rewards = []

    for t in range(nb_timesteps):  # iterate over time steps
        env.render()  # display the environment
        state, reward, done, info = env.step(policy(t))  # implement the action chosen by the policy
        #print(t,reward)
        rewards.append(reward)  # add 1 to the rewards list

        if done:  # the episode ends either if the pole is > 15 deg from vertical or the cart move by > 2.4 unit from the centre
            cumulative_reward = sum(rewards)
            print("episode {} finished after {} timesteps. Total reward: {}".format(episode, t + 1, cumulative_reward))
            break

env.close()