from gym_env import OrekitEnv
import pandas as pd
import matplotlib.pyplot as plt
import os 


# initial_state = [1000*1e3, 0.01, 0.01, 0.01, 0.01, 0.01]
# target_state = [6300*1e3, 0.23, 5.3, 24.0, 24.0, 10.0]
# simulation_date = [2018, 2, 16, 12, 10, 0.0]
# simulation_duration = 24.0 * 60.0 ** 2 * 4
# spacecraft_mass = [500.0, 150.0]
# timestep = 53*60+44
# env = OrekitEnv(initial_state, target_state, simulation_date, simulation_duration, spacecraft_mass, timestep)


folder = 'results/episode_stats'
plt.show()
files = os.listdir(folder)
for file in files:
    path = os.path.join(folder, file)
    df = pd.read_csv(path)
    # df = df[df.iloc[:,0] < 5000]
    filtered_df = df[(df.iloc[:,1] > -1000) & (df.iloc[:,1] < 10000)]
    episode = df.iloc[:,0]
    reward = df.iloc[:,1]
    fuel = df.iloc[:,2]
    distance = df.iloc[:,3]
    initial_distance = df.iloc[:,5] if df.shape[1] >= 6 else None

    filtered_episode = filtered_df.iloc[:,0]
    filtered_reward = filtered_df.iloc[:,1]
    filtered_fuel = filtered_df.iloc[:,2]
    filtered_distance = filtered_df.iloc[:,3]

    plt.title(file)
    plt.plot(filtered_episode, filtered_reward)
    plt.show()
    
    plt.title(file)
    plt.plot(episode, distance)
    plt.axhline(y=0.18385840583221036, color='red')
    plt.show()

exit()

# df = pd.read_csv('results/episode_stats/distance_only_discrete_4964.csv')
# # data = pd.read_csv('results/episode_stats/distance_only_discrete_4964.csv')
# df = df[df.iloc[:,0] < 5000]
# filtered_df = df[(df.iloc[:,1] > -1000) & (df.iloc[:,1] < 10000)]
# episode = df.iloc[:,0]
# reward = df.iloc[:,1]
# fuel = df.iloc[:,2]
# distance = df.iloc[:,3]
# initial_distance = df.iloc[:,5] if df.shape[1] >= 6 else None

# filtered_episode = filtered_df.iloc[:,0]
# filtered_reward = filtered_df.iloc[:,1]
# filtered_fuel = filtered_df.iloc[:,2]
# filtered_distance = filtered_df.iloc[:,3]

# plt.show()

# plt.title('Discrete Values Model Structure Reward')
# plt.plot(filtered_episode, filtered_reward)
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.show()

# plt.title('Discrete Values Model Distance From Target')
# plt.plot(episode, distance)
# plt.axhline(y=0.18385840583221036, color='red')
# # plt.plot(episode, initial_distance, color='red')
# plt.xlabel('Episode')
# plt.ylabel('Distance')
# plt.show()


# basic discrete: 3
# less penalty: 2
# no action: 1
# new reward
# new reward no fuel penalty: 1


# newer reward

# retrain:
# baseline
# discrete newer reward
# discrete new reward