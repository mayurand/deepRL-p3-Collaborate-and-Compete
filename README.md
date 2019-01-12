# Udacity Deep Reinforcement Nanodegree Project 3: Collaboration and Competition
This repository contains implementation of Collaboration and Competition project as a part of Udacity's Deep Reinforcement Learning Nanodegree program.

In this project a two player game of tennis is played where the control of rackets is learned to bounce ball over a net. The [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment from the Unity ML Agents Toolkit is used here.
<br> 
![Tennis](images/tennis.gif)
<br>
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically, after each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started

### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Install Anaconda for Python3 from [here](https://www.anaconda.com/download).

2. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name p3_drlnd python=3.6
	source activate p3_drlnd
	```
	- __Windows__: 
	```bash
	conda create --name p3_drlnd python=3.6 
	activate p3_drlnd
	```
	
3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.
	
4. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/mayurand/deepRL-p3-Collaborate-and-Compete.git
cd deepRL-p3-Collaborate-and-Compete/python
pip install .
```

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `p3_drlnd` environment.
```bash
python -m ipykernel install --user --name p3_drlnd --display-name "p3_drlnd"
```

### Environment setup

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

3. Test if the environment is correctly installed:
```bash
cd deepRL-p3-Collaborate-and-Compete/p3_collab-compet #navigate to the p3_collab-compet directory
source activate p3_drlnd  #Activate the python environment
jupyter notebook
```
4. Open the `Test_the_environment.ipynb` and run the cells with SHIFT+ENTER. If the environment is correctly installed, you should get to see the Unity environment in another window and values for state and action spaces under `2. Examine the State and Action Spaces`. 


### Train the agent or run a trained agent
1. To train an agent for the above environment:
```bash
cd deepRL-p3-Collaborate-and-Compete/p3_collab-compet #navigate to the p2_continuous-control directory
source activate p3_drlnd  #Activate the python environment
jupyter notebook
```
2. Open the `Tennis_MADDPG.ipynb` and run the cells with SHIFT+ENTER. 

3. To directly run the trained model, navigate to the `6. Watch Smart Agents Play!` in the notebook and run the code.

__Note:__ Before running code in a notebook, change the kernel to match the `p3_drlnd` environment by using the drop-down `Kernel` menu. 


### Report
See the [report](https://github.com/mayurand/deepRL-p3-Collaborate-and-Compete/Report.ipynb) for more details on the implementation.

