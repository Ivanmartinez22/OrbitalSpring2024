# from gym_env import OrekitEnv


# initial_state = [1000*1e3, 0.01, 0.01, 0.01, 0.01, 0.01]
# target_state = [6300*1e3, 0.23, 5.3, 24.0, 24.0, 10.0]
# simulation_date = [2018, 2, 16, 12, 10, 0.0]
# simulation_duration = 24.0 * 60.0 ** 2 * 4
# spacecraft_mass = [500.0, 150.0]
# timestep = 53*60+44
# env = OrekitEnv(initial_state, target_state, simulation_date, simulation_duration, spacecraft_mass, timestep)

# env.step([0, 24.064152, 0])


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

data = pd.read_csv('results/episode_stats/12353_TD3.csv')
# data = pd.read_csv('results/episode_stats/distance_only_discrete_4964.csv')
data = data[data.iloc[:,1] > -100000]
episode = data.iloc[:,0]
reward = data.iloc[:,1]
fuel = data.iloc[:,2]
distance = data.iloc[:,3]
# n_hits = (distance < 0.15).astype(int).sum()
# print(n_hits)
plt.figure(figsize=(8, 6))  # Optional: Adjust the figure size
plt.plot(episode, reward)
plt.title('reward')
plt.show()
plt.plot(episode, distance)
plt.title('final distance')
plt.show()
plt.plot(episode, fuel)
plt.title('fuel remaining')
plt.show()


# basic discrete: 3
# less penalty: 2
# no action: 1
# new reward
# new reward no fuel penalty: 1