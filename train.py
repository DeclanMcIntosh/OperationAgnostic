from datetime import datetime, timedelta
import json
import time

import gym
import matplotlib.pyplot as plt
import names
import numpy as np
from gym.envs.registration import spec

from agent import *
from operations import *
from utils import *


def train(configString):
    with open(configString) as f:
        data  = f.read()
    config = json.loads(data)
    print_config(config)

    # seed random, we only use the random but the gym environments also use the numpy random
    r.seed(config['seed'])
    np.random.seed(config['seed'])

    config['action_handler'] = dispatcher[config['action_handler']]
    operations = []
    for operation in config['operations']:
        operations.append(operations_dispatcher[operation])
    config['operations'] = operations

    inno = Innovation(config['inputs']+config['outputs'])
    agentParams=dict(inputs=config['inputs'], outputs=config['outputs'], max_layers=config['max_layers'], default_output=config['default_output'], operations=config['operations'], 
                add_node_rate=config['add_node_rate'], add_connection_rate=config['add_connection_rate'], remove_node_rate=config['remove_node_rate'], remove_connection_rate=config['remove_connection_rate'])

    env = gym.make(config['envName'])
    env.reset()
    env.seed(r.randint(0,999999))

    # intiallize agents and mutate all agents 
    agents = []
    species = []
    fitnesses = []
    startingSpeciesName = names.get_full_name(gender='female')
    for x in range(config['population']):
        agents.append(AgentFFO(**agentParams))
        species.append(startingSpeciesName)
        fitnesses.append(0)
    for agent in agents:
        agent.mutate(inno)

    fitnessHistory = []
    speciesNumberHistory = []

    print(f'Starting training at {datetime.now()}...\n')
    print('Generation \t# species \tMean fitness \tMax fitness \tElapsed time')
    print('--------------------------------------------------------------------------------')

    for ___ in range(config['generations']):
        time_generation = time.time()
        # play all agents in environment
        for x in range(config['population']):
            obs = env.reset()
            env.seed(r.randint(0,999999))
            _ = 0
            terminal = False
            fitness = 0
            for n in range(config['numberOfTrialsToRun']):
                while _ < config['maxStepsPerRun'] and not terminal:
                    _ +=1   
                    action = config['action_handler'](obs, agents[x])
                    obs, rew, terminal, info = env.step(action)
                    #env.render()
                    fitness += rew
                terminal = False
                _ =0
                obs = env.reset()
                env.seed(r.randint(0,999999))
            #print(fitness)
            fitnesses[x] = fitness/config['numberOfTrialsToRun']
            #print(x)

        print(___, end='\t\t') # Done evaluating all agents for generation

        # speciate agents by sortting by fitness then deciding on species then 
        
        zipped = zip(fitnesses, agents, species)
        
        zipped = sorted(zipped, key = lambda x: x[0])
        zipped.reverse()
        fitnesses, agents, species = zip(*zipped)
        fitnesses, agents, species = list(fitnesses), list(agents), list(species)
        representatives = {}
        for x in range(config['population']):
            match = None
            genome = agents[x].getGenome()
            for representative in representatives.keys():
                correspondace = IoU(genome, representatives[representative])
                #print(correspondace)
                if correspondace > config['compatabilityThreashold']:
                    match = representative
            if not match is None:
                species[x] = match
            else: # if no match  
                if species[x] in representatives.keys(): 
                    species[x] = names.get_full_name(gender='female')
                representatives[species[x]] = genome

        print(len(representatives.keys()), end='\t\t') # number of species

        speciesFitnesses = representatives.copy()
        for key in speciesFitnesses.keys(): speciesFitnesses[key] = 0
        for x in range(config['population']): speciesFitnesses[species[x]] += fitnesses[x]
        for key in speciesFitnesses.keys(): speciesFitnesses[key] = speciesFitnesses[key]/ species.count(key) 

        meanFitness = np.mean(np.array((list(speciesFitnesses.values()))))
        maxFitness  = np.max(np.array((list(speciesFitnesses.values()))))
        adjustedSpeciesFitnesses = speciesFitnesses.copy()
        for key in adjustedSpeciesFitnesses.keys(): adjustedSpeciesFitnesses[key] = adjustedSpeciesFitnesses[key] - meanFitness
        totalFitnesses = np.sum(np.array((list(speciesFitnesses.values()))))
        for key in adjustedSpeciesFitnesses.keys(): adjustedSpeciesFitnesses[key] = adjustedSpeciesFitnesses[key] / totalFitnesses

        for x in range(int(config['population']*config['populationOverwriteRate'])):
            # Pick species
            speciesToOverwrite = r.choice(list(adjustedSpeciesFitnesses.keys()))
            speciesToPropegate = r.choices(list(adjustedSpeciesFitnesses.keys()), weights=list(adjustedSpeciesFitnesses.values()))[0]

            # Pick worst agent to overwrite and random agent to propegate
            agentToOverwriteIndex = [i for i, k in enumerate(species) if k == speciesToOverwrite]
            agentToOverwriteIndex = agentToOverwriteIndex[-1]
            agentToPropegateIndex = r.choice([i for i, k in enumerate(species) if k == speciesToPropegate])

            # Make copy to replace bad agent
            agents[agentToOverwriteIndex] = AgentFFO(**agentParams)
            agents[agentToOverwriteIndex].crossoverModels(agents[agentToPropegateIndex])
            species[agentToOverwriteIndex] = species[agentToPropegateIndex]
            fitnesses[agentToOverwriteIndex] = fitnesses[agentToPropegateIndex]

            # Remove extinct species
            if species.count(speciesToOverwrite) == 0:
                del adjustedSpeciesFitnesses[speciesToOverwrite]

        # Mutate all the non best agents 
        speciesRefs = list(adjustedSpeciesFitnesses.keys())
        for x in range(len(speciesRefs)):
            speciesIndexs = [i for i, k in enumerate(species) if k == speciesRefs[x]]
            for index in speciesIndexs[config['topXtoSave']:]:
                for x in range(r.randint(1,config['MaxMutations'])):
                    agents[index].mutate(inno)

        zipped = zip(fitnesses, agents, species)
        
        zipped = sorted(zipped, key = lambda x: x[0])
        zipped.reverse()
        fitnesses, agents, species = zip(*zipped)
        fitnesses, agents, species = list(fitnesses), list(agents), lis            speciesIndexes = [i for i, k in enumerate(species) if k == speciesRefs[x]]
            for index in speciesIndexes[config['topXtoSave']:]:
                agents[index].mutate(inno)t(species)

        for x in range(len(speciesRefs)):
            speciesIndexes = [i for i, k in enumerate(species) if k == speciesRefs[x]]
            for index in speciesIndexes[config['topXtoSave']:]:
                agents[index].mutate(inno)

        print(f'{meanFitness:.2f}', end='\t\t') # average fitness
        print(f'{maxFitness:.2f}', end='\t\t')
        print(str(timedelta(seconds=time.time() - time_generation))) # elapsed time
        fitnessHistory.append(meanFitness)
        speciesNumberHistory.append(len(representatives.keys()))

    # visualize best model
    agents[0].VisualizeModel()

    # create figure and axis objects with subplots()
    fig,ax = plt.subplots()
    # make a plot
    ax.plot(fitnessHistory, color="red", marker="o")
    # set x-axis label
    ax.set_xlabel("Generation",fontsize=14)
    # set y-axis label
    ax.set_ylabel("Fitness",color="red",fontsize=14)
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(speciesNumberHistory,color="blue",marker="o")
    ax2.set_ylabel("# of Species",color="blue",fontsize=14)
    plt.show()


def print_config(config):
    print('***********************************************************')
    print(f'inputs: {config["inputs"]}')
    print(f'outputs: {config["outputs"]}')
    print(f'max_layers: {config["max_layers"]}')
    print(f'populations: {config["population"]}')
    print(f'generations: {config["generations"]}')
    print('***********************************************************')
    print()
    print()

if __name__ == '__main__':
    training_start = time.time()

    #configString = 'configs/config_cart_pole_binary_nand.json'
    #configString = 'configs/config_cart_pole_binary_nand_binned.json'
    configString = 'configs/config_bipedal_walker_nand_binned.json'
    #configString = 'configs/config_cart_pole_uint8_add.json'
    #configString = 'configs/config_lunar_lander_uint8_add.json'
    train(configString)

    training_end = time.time()
    training_time = training_end - training_start
    print("Total train time: {}".format(str(timedelta(seconds=training_time))))
