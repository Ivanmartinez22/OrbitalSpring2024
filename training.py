# #############################################################################
# ## Restricted Rights
# ## WARNING: This is a restricted distribution L3HARRIS REPOSITORY file.
# ##          Do Not Use Under A Government Charge Number Without Permission.
# #############################################################################

# *****************************************************************************
# FILE:             stationkeeping.py
#
#    Copyright (C) 2024 L3Harris Technologies, Inc.  All rights reserved.
#
# CLASSIFICATION:   Unclassified
#
# DESCRIPTION:
#  Kicks off RL-based orbital trajectory calculator to generate optimal path 
#  from initial orbit/state to goal orbit/state
#
# LIMITATIONS:
#  [TODO: Describe limitations of the code contained in this file.]
#
# SOFTWARE HISTORY:
#  06FEB24 90PA PTR#MISSANGR-01026  P. Sridhar
#               Initial coding.
# *****************************************************************************

import sys
import torch
import numpy as np
from stable_baselines3 import TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from gym_env import OrekitEnv

print('GPU found') if torch.cuda.is_available() else print('GPU not found')

def train_model(alg, initial_state, target_state, simulation_date, 
                simulation_duration, spacecraft_mass, simulation_stepT):

   # Create environment instance
   env = OrekitEnv(initial_state, target_state, simulation_date, 
                   simulation_duration, spacecraft_mass, simulation_stepT)
   # Get action space from environment
   n_actions = env.action_space.shape[-1]
   # Define the action noise (continuous action space)
   action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

   if(alg == "DDPG"):
      # Create the TD3 model
      model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, device="auto", tau=0.01)
   elif(alg == "TD3"):
      model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, device="auto", tau=0.01)
   else:
      print("Unknown model, check again and run")
      sys.exit()

   env.alg = alg

   # Options for loading existing model
   # model = DDPG.load("ddpg_model", device="cpu")
   # model.set_env(env)

   # Train & save model
   model.learn(total_timesteps=415000, log_interval=10)
   model.save('models/'+str(env.id)+"_"+ alg +"_model")

   # Generate .txt of reward/episode trained
   env.write_reward()
