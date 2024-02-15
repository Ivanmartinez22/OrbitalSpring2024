# Optimal Policy for Maneuers (OPM) Integration Test

## Software dependencies
- Python 3.11.0 (see requirements.txt for software package versions)

## Pre-Tool Setup for Windows
- Install Anaconda
   - https://www.anaconda.com/download
   - This installs anaconda/python 
- Install Java  
   - run powershell as admin use following command 
      winget install Oracle.JDK.21
   - download java jdk with correct binary for windows from here https://www.oracle.com/java/technologies/javase/jdk21-archive-downloads.html
   - Ensure you have the following sytem environment variables in path:
      C:\Program Files\Common Files\Oracle\Java\javapath
      C:\Program Files\Java\jdk-21
- Luanch Anaconda Navigator
   - Launch Visual Studio Code from Anaconda this will allow Visual Studio Code Terminal to have access to the Anaconda Environment (I have not been able to get normally launched vscode or powershell to properly access Anaconda yet)



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

## Execution facts so far 
- Runs very slow I have not been able to see end of execution 

## References

A Reinforcement Learning Approach to Spacecraft Trajectory Optimization, [Daniel S. Kolosa](https://github.com/dkolosa/Satmind).