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
## To Train On GPU (optional)
- Install CUDA using the command: `conda install cuda -c nvidia`
   - If using an older graphics card, you may need to install an older version of cuda with the following command: `conda install cuda -c nvidia/label/cuda-VERSION`
   - To see what version of CUDA you need, check these the second table to find your graphics card, then use the table above to find the CUDA SDK version your GPU can run: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
- Install pytorch with conda support with the following command: `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
   - This command may vary depending on updates and the CUDA version installed, a generator for the command can be found at the pytorch website: https://pytorch.org/
- NOTE: Older versions of CUDA and pytorch-cuda may create inconsistencies with packages in the requirements.txt
- The program should now print "GPU found" when running main.py. If not, check to make sure your graphics card drivers are updated

## Analysis Execution (main.py)
- Define initial state, target state, simulation_date, simulation_duration, spacecraft_mass, and simulation_stepT
- Run **main.py** by passing in either **DDPG** or **TD3** to the **stationkeeping** function
   ## GUI usage (gui.py)
   - To use the GUI, run **gui.py**
   - While in the GUI, you can edit the values of the attributes to run the program based on your own values
   - To run the program, press **run** and the program will begin to execute in a new window that displays the command line output
         - To see the full output, you must answer the prompts for the live visuzalier and choose training/predict in the command line
         - To see the visualizer, press **visualize** to see the original visualizer run within the GUI window

## Results
- Successful termination states during training are found in the **results** directory
- A zip of the trained model will appear in the **models** directory
- Change **line 39** of **visual_orbit.py** to whatever file within the *results/state* folder you want to visualize
- Run **visual_orbit.py**

## Execution facts so far 
- Runs very slow I have not been able to see end of execution 

## References

A Reinforcement Learning Approach to Spacecraft Trajectory Optimization, [Daniel S. Kolosa](https://github.com/dkolosa/Satmind).