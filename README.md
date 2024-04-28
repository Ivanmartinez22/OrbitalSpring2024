# Optimal Policy for Maneuvers (OPM)

## Overview
Traditional satellite maneuvering methods often rely on pre-programmed commands or human intervention. The developed tool addresses these limitations by training an agent, in the role of a satellite, to make intelligent decisions in real-time to optimize its trajectory and orbital path for various objectives. This tool also develops a novel approach to the way satellite trajectory changes are performed in that it utilizes Reinforcement Learning (RL) to develop continuous and high precision orbital transfer actions, as opposed to fewer, larger impulse burns (e.g. Hohmann Transfer)

## Algorithm Background
Twin Delayed Deep Deterministic Policy Gradient (TD3)

- Direct successor of Deep Deterministic Policy Gradient (DDPG) algorithm

- Clipped double Q-Learning

- Delayed policy update

- Target policy smoothing

## Analysis Assumptions
- Consider it as the restricted two-body problem between Earth and the spacecraft

- Spacecraft set in Earth-centered interial frame (ECI)

- Utilizes a Runge-Kutta numerical propagator to simulate motion

## Future Work

Opportunities to further improve this tool include the implementation of a more complex maneuvers and actions, shifting from a continuous to a discrete action space, and having the RL agent be able to accomodate for either a multi-agent training situation or a dynamic goal state

GUI could have the newer features from main integrated into it such as retraining existing models. 

## Contributions and Acknowledgments

This tool was initially developed by Texas A&M University Computer Science students, as part of L3Harris' Capstone partnership with Texas A&M University, as facilitated through the Mission Analysis Group (MAG). Underlying research and comparison work was drawn from the following paper: *A Reinforcement Learning Approach to Spacecraft Trajectory Optimization, [Daniel S. Kolosa](https://github.com/dkolosa/Satmind).*

## License

Copyright (C) 2024 L3Harris Technologies, Inc.  All rights reserved.

## Contact

For inquiries or support, please contact MAG Members Pranav Sridhar, Kyle Casey, or Josef Zapletal
