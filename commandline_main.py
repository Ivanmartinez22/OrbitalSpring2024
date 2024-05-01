# #############################################################################
# ## Restricted Rights
# ## WARNING: This is a restricted distribution L3HARRIS REPOSITORY file.
# ##          Do Not Use Under A Government Charge Number Without Permission.
# #############################################################################

# *****************************************************************************
# FILE:             commandline_main.py
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

from testing.env_test import checkPackageVersion, testEnvironment
from training import train_model, load_model, predict, retrain_model
import os


testEnvironment()

# Page 35 RP(A Reinforcement Learning Approach to Spacecraft Trajectory);
# [a(m), e, i(deg), omega/w(deg), Omega/raan(deg), TrueAnomaly(v)]
initial_state = [5500*1e3, 0.20, 5.0, 20.0, 20.0, 10.0]
target_state = [5800*1e3, 0.21, 5.1, 20.5, 20.5, 10.0]
# target_state = [6300*1e3, 0.23, 5.3, 24.0, 24.0, 10.0]
simulation_date = [2018, 2, 16, 12, 10, 0.0]
simulation_duration = 24.0 * 60.0 ** 2 * 4
spacecraft_mass = [500.0, 150.0]
# Let Spacecraft take an action every (step) amount of seconds
simulation_stepT = 500.0
visualize = True

# print("""

# Live Visualization Options:
# - Type 'y' to turn on live visualizer for predictions and training.
# - Type 'n' to disable live visualization.
      
# Options:
# 1. Predict with an existing model
# 2. Train a new model
# 3. Retrain an existing model

# To select an option, enter the corresponding number in the command line:

# Example Usage:
# Turn on Live visualizer for predictions and training (y/n): y
# Enter 1 for predict, Enter 2 for training, Enter 3 for retraining existing model: 2
# Enter algorithm: PPO, TD3, DDPG: PPO
# """)

user_viz_in = input("Turn on Live visualizer for predictions and training (y/n): ")
user_train_predict_in = input("Enter 1 for predict, Enter 2 for training, Enter 3 for retraining existing model: ")
visualize = False
if user_viz_in == "y":
    visualize = True

if user_train_predict_in == "1":
    alg = input("Enter algorithm: PPO, TD3, DDPG: ")
    model_name_input = input("Enter model name: ")
    model_name = "models/" + model_name_input
    model = load_model(alg, model_name)
    predict(model, initial_state, target_state, simulation_date, 
                simulation_duration, spacecraft_mass, simulation_stepT, visualize)
elif user_train_predict_in == "3":
    alg = input("Enter algorithm: PPO, TD3, DDPG: ")
    model_name_input = input("Enter input model name: ")
    export_model_name_input = input("Enter export model name: ")
    retrain_model(alg, initial_state, target_state, simulation_date, simulation_duration, spacecraft_mass, simulation_stepT, visualize, model_name_input, export_model_name_input)
elif user_train_predict_in == "2":
    alg = input("Enter algorithm: PPO, TD3, DDPG: ")
    train_model(alg, initial_state, target_state, simulation_date, 
                simulation_duration, spacecraft_mass, simulation_stepT, visualize)
