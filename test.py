from gym_env import OrekitEnv


initial_state = [1000*1e3, 0.01, 0.01, 0.01, 0.01, 0.01]
target_state = [6300*1e3, 0.23, 5.3, 24.0, 24.0, 10.0]
simulation_date = [2018, 2, 16, 12, 10, 0.0]
simulation_duration = 24.0 * 60.0 ** 2 * 4
spacecraft_mass = [500.0, 150.0]
timestep = 53*60+44
env = OrekitEnv(initial_state, target_state, simulation_date, simulation_duration, spacecraft_mass, timestep)

env.step([0, 24.064152, 0])