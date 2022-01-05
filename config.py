import gym
from gym.envs.registration import spec
import gym_cartpole_swingup

from agent import * 
from utils import *
from operations import *

# Set parameters for agents
inputs = 4
outputs = 1

agentParams=dict(inputs=inputs, outputs=outputs, max_layers=4, default_output=0, operations=[eightBitAdd], 
                add_node_rate=0.25, add_connection_rate=0.5, remove_node_rate=0.1, remove_connection_rate=0.1)
# Set parameters for training 
population = 64
generations = 128
maxStepsPerRun = 100
compatabilityThreashold = 0.5
saveBestModelEachSpecies = True
populationOverwriteRate = 0.125
numberOfTrialsToRun = 1
topXtoSave = 1

inno = Innovation(inputs+outputs)

# Set parameters for the environment 
env=gym.make('CartPole-v0')

action_handler = CartPole_Action