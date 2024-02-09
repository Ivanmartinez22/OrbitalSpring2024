# Optimal Policy for Maneuers (OPM) Integration Test

## Software dependencies
- Python 3.11.0 (see requirements.txt for software package versions)

## Tool Setup
- Create a virtual environment using conda
   - Run the following command in command prompt: `conda create -n "virtual environment name" python=3.11.0`
- Activate the virtual environment with this command
   - `conda activate "virtual environment name"`
- Install the main packages using pip with the following command in command prompt: `pip install -r "path/to/requirements.txt"`
- Install Orekit using conda with the following command in command prompt: `conda install conda-forge::orekit`

## Analysis Execution (main.py)
- Define initial state, target state, simulation_date, simulation_duration, spacecraft_mass, and simulation_stepT
- Run **main.py** by passing in either **DDPG** or **TD3** to the **stationkeeping** function

## Results
- Successful termination states during training are found in the **results** directory
- A zip of the trained model will appear in the **models** directory
- Change **line 39** of **visual_orbit.py** to whatever file within the *results/state* folder you want to visualize
- Run **visual_orbit.py**

## References

A Reinforcement Learning Approach to Spacecraft Trajectory Optimization, [Daniel S. Kolosa](https://github.com/dkolosa/Satmind).