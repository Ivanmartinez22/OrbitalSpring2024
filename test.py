from gym_env import OrekitEnv
import pandas as pd
import matplotlib.pyplot as plt


# initial_state = [1000*1e3, 0.01, 0.01, 0.01, 0.01, 0.01]
# target_state = [6300*1e3, 0.23, 5.3, 24.0, 24.0, 10.0]
# simulation_date = [2018, 2, 16, 12, 10, 0.0]
# simulation_duration = 24.0 * 60.0 ** 2 * 4
# spacecraft_mass = [500.0, 150.0]
# timestep = 53*60+44
# env = OrekitEnv(initial_state, target_state, simulation_date, simulation_duration, spacecraft_mass, timestep)

# data = pd.read_csv('results/episode_stats/continuous_earth.csv')
# episode = data.iloc[:,0]
# reward = data.iloc[:,1]
# fuel = data.iloc[:,2]
# distance = data.iloc[:,3]
# n_hits = (distance < 0.15).astype(int).sum()
# print(n_hits)
# plt.figure(figsize=(8, 6))  # Optional: Adjust the figure size
# plt.plot(episode, reward)
# plt.show()
# plt.plot(episode, distance)
# plt.plot(episode, fuel)
# plt.show()

df = pd.read_csv('results/episode_stats/impulse_1.csv')
# data = pd.read_csv('results/episode_stats/distance_only_discrete_4964.csv')
df = df[df.iloc[:,0] < 5000]
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

plt.show()

plt.title('Action Decision Model Structure Reward')
plt.plot(filtered_episode, filtered_reward)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

plt.title('Action Decision Model Structure Distance From Target')
plt.plot(episode, distance)
# plt.axhline(y=0.18385840583221036, color='red')
plt.plot(episode, initial_distance, color='red')
plt.xlabel('Episode')
plt.ylabel('Distance')
plt.show()