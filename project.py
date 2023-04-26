import numpy as np
import gym # pip intall gym
import random
import matplotlib.pyplot as plt

from gym.wrappers import RecordEpisodeStatistics

env = gym.make('FrozenLake-v1', is_slippery = False, render_mode="human")

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

qtable = np.zeros((state_space_size, action_space_size))


#Hyper Parameters

total_episodes = 10000
learning_rate = 0.2 # 0-1
max_steps = 100
gamma = 0.99

epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay_rate = 0.001

rewards = []
avg_rewards = []  # List to store average score obtained every 100 episodes
episode_rewards = 0

for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        if random.uniform(0, 1) > epsilon:
            try:
                action = np.argmax(qtable[state, :])  # Exploit
            except:
                action = np.argmax(qtable[state[0], :])  # Exploit
        else:
            action = env.action_space.sample()  # Explore

        new_state, reward, done, info, prob = env.step(action)

        max_new_state = np.max(qtable[new_state, :])

        try:
            qtable[state[0], action] = qtable[state[0], action] + learning_rate * (
                    reward + gamma * max_new_state - qtable[state[0], action])
        except:
            qtable[state, action] = qtable[state, action] + learning_rate * (
                    reward + gamma * max_new_state - qtable[state, action])

        total_rewards += reward
        episode_rewards += reward

        state = new_state
        if done:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)
    if episode % 100 == 0:
        avg_reward = np.mean(rewards[-100:])
        avg_rewards.append(avg_reward)
        episode_rewards = 0

print(avg_rewards)
print("Average score:", str(sum(rewards) / total_episodes))

# Plot the average score obtained every 100 episodes
plt.plot(range(0, total_episodes, 100), avg_rewards)
plt.xlabel("Number of episodes")
plt.ylabel("Average score")
plt.title("Learning progress")
plt.show()
        


env = gym.make('FrozenLake-v1', is_slippery = False, render_mode="human")
# env.render_mode='human'
env.reset()
for episode in range(100):
    state = env.reset()
    print(env.action_space.n)
    step = 0
    done = False
    
    print("Episode:", episode+1)
    
    for step in range(max_steps):
        try:
            action = np.argmax(qtable[state,:])
        except:
            action = np.argmax(qtable[state[0],:])
        new_state, reward, done, info, _ = env.step(action)
        # env.render(mode='human')
        if done:
            
            print("Number of Steps:", step)
            break
        state = new_state
        
env.close()