# #############################################################################
# ## Restricted Rights
# ## WARNING: This is a restricted distribution L3HARRIS REPOSITORY file.
# ##          Do Not Use Under A Government Charge Number Without Permission.
# #############################################################################

# *****************************************************************************
# FILE:             main.py
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
from training import train_model, load_model, predict

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

# model = load_model('PPO', 'models/29652_PPO_model')
# predict(model, initial_state, target_state, simulation_date, 
#             simulation_duration, spacecraft_mass, simulation_stepT, visualize)

train_model("PPO", initial_state, target_state, simulation_date, 
            simulation_duration, spacecraft_mass, simulation_stepT, False)