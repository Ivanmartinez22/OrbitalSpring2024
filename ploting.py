import pandas as pd
import matplotlib.pyplot as plt
import os 


# folder = 'results/episode_stats'
# plt.show()
# files = os.listdir(folder)
# for file in files:
#     path = os.path.join(folder, file)
#     df = pd.read_csv(path)
#     # df = df[df.iloc[:,0] < 5000]
#     filtered_df = df[(df.iloc[:,1] > -1000) & (df.iloc[:,1] < 10000)]
#     episode = df.iloc[:,0]
#     reward = df.iloc[:,1]
#     fuel = df.iloc[:,2]
#     distance = df.iloc[:,3]
#     initial_distance = df.iloc[:,5] if df.shape[1] >= 6 else None

#     filtered_episode = filtered_df.iloc[:,0]
#     filtered_reward = filtered_df.iloc[:,1]
#     filtered_fuel = filtered_df.iloc[:,2]
#     filtered_distance = filtered_df.iloc[:,3]

#     plt.title(file)
#     plt.plot(filtered_episode, filtered_reward)
#     plt.show()
    
#     plt.title(file)
#     plt.plot(episode, distance)
#     plt.axhline(y=0.18385840583221036, color='red')
#     plt.show()

# exit()

df = pd.read_csv('results/episode_stats/final_model.csv')
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

plt.title('Current Model Reward')
plt.plot(filtered_episode, filtered_reward)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

plt.title('Current Model Distance From Target')
plt.plot(episode, distance)
plt.axhline(y=0.18385840583221036, color='red')
plt.xlabel('Episode')
plt.ylabel('Distance')
plt.show()
