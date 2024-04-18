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
from training import train_model

testEnvironment()

# # # Page 35 RP(A Reinforcement Learning Approach to Spacecraft Trajectory);
# # # [a(m), e, i(deg), omega/w(deg), Omega/raan(deg), TrueAnomaly(v)]
# # initial_state = [5500*1e3, 0.20,5.0, 20.0, 20.0,10.0]
# # target_state = [6300*1e3, 0.23, 5.3, 24.0, 24.0, 10.0]
# # simulation_date = [2018, 2, 16, 12, 10, 0.0]
# # simulation_duration = 24.0 * 60.0 ** 2 * 4
# # spacecraft_mass = [500.0, 150.0]
# # # Let Spacecraft take an action every (step) amount of seconds
# # simulation_stepT = 500.0


# # Saved Data is in format:
# '''
# data_set_#
# coordinates
# intial, a, e, i, omega, w, v
# final, a, e, i, omega, w, v
# duration, max_thrust, max_speed
# date_1, date_2, date_3, date_4, date_5, date_6
# mass_1, mass_2
# '''

# file_path = "saved_data.txt"

# def parse_data(file_path):
#     parsed_data = {
#         "data_set_number": None,
#         "coordinate_type": None,
#         "coordinates": {
#             "initial": {},
#             "final": {}
#         },
#         "stats": {
#             "duration": None,
#             "max_thrust": None,
#             "max_speed": None
#         },
#         "date": [],
#         "masses": []
#     }

#     with open(file_path, 'r') as file:
#         lines = file.readlines()

#     dataset_num = input("Enter Which Dataset to Use: ")
#     foundDataset = False

#     print("LINES: ", lines)
    
#     for i, line in enumerate(lines):
#         parts = line.strip().split(', ')
        
#         while foundDataset == False:
#             continue

#         if parts[0] == dataset_num:
#             parsed_data["data_set_number"] = parts[0]
#             foundDataset = True
#         elif parts[0] in ['keplerian','cartesian']:
#             parsed_data["coordinate_type"] = line
#         elif parts[0] in ['initial', 'final']:
#             parsed_data["coordinates"][parts[0]] = {
#                 "a": float(eval(parts[1])),
#                 "e": float(parts[2]),
#                 "i": float(parts[3]),
#                 "omega": float(parts[4]),
#                 "w": float(parts[5]),
#                 "v": float(parts[6])
#             }
#         elif parts[0] == 'results':
#             parsed_data["stats"] = {
#                 "duration": eval(parts[1]),
#                 "max_thrust": float(parts[2]),
#                 "max_speed": float(parts[3])
#             }
#         elif parts[0] == 'date':
#             parsed_data["date"] = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), float(parts[6])]
#         elif parts[0] == 'mass':
#             parsed_data["masses"] = [float(parts[1]), float(parts[2])]
    
#     return parsed_data

# parsed_data = parse_data(file_path)

# parsed_data = parse_data('saved_data.txt')
# # print("STATS: ", parsed_data["date"])
# # train_model("TD3", parsed_data["coordinates"]["initial"], parsed_data["coordinates"]["final"],
# #             parsed_data["stats"]["duration"],parsed_data["stats"]["duration"], 
# #             parsed_data["masses"], simulation_stepT)

# initial_state = list((parsed_data["coordinates"]["initial"]).values())
# target_state = list((parsed_data["coordinates"]["final"]).values())
# simulation_date = parsed_data["date"]
# simulation_duration = parsed_data["stats"]["duration"]
# spacecraft_mass = parsed_data["masses"]
# simulation_stepT = 500.0

initial_state = [5500*1e3, 0.20,5.0, 20.0, 20.0, 10.0]
target_state = [6300*1e3, 0.23, 5.3, 24.0, 24.0, 10.0]
simulation_date = [2018, 2, 16, 12, 10, 0.0]
simulation_duration = 24.0 * 60.0 ** 2 * 4
spacecraft_mass = [500.0, 150.0]
# Let Spacecraft take an action every (step) amount of seconds
simulation_stepT = 500.0
# print(target_state)
# print(list((parsed_data["coordinates"]["final"]).values()))

train_model("TD3", initial_state, target_state, simulation_date, 
            simulation_duration, spacecraft_mass, simulation_stepT)