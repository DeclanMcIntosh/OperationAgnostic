import gym
from gym.envs.registration import spec
import gym_cartpole_swingup
import names
from agent import * 
import numpy as np
import matplotlib.pyplot as plt

from operations import *
from utils import *
from config import *

if __name__ == '__main__':    
    env.reset()

    # intiallize agents and mutate all agents 
    agents = []
    species = []
    fitnesses = []
    startingSpeciesName = names.get_full_name(gender='female')
    for x in range(population):
        agents.append(AgentFFO(**agentParams))
        species.append(startingSpeciesName)
        fitnesses.append(0)
    for agent in agents:
        agent.mutate(inno)

    fitnessHistory = []
    speciesNumberHistory = []

    for ___ in range(generations):
        # play all agents in environment
        for x in range(population):
            obs = env.reset()
            _ = 0
            terminal = False
            fitness = 0
            for n in range(numberOfTrialsToRun):
                while _ < maxStepsPerRun and not terminal:
                    _ +=1   
                    action = action_handler(obs, agents[x])
                    obs, rew, terminal, info = env.step(action)
                    #env.render()
                    fitness += rew
            #print(fitness)
            fitnesses[x] = fitness
            #print(x)

        print('Done Evaluating All Agents for generation: ' + str(___))
        # speciate agents by sortting by fitness then deciding on species then 
        
        zipped = zip(fitnesses, agents, species)
        
        zipped = sorted(zipped, key = lambda x: x[0])
        zipped.reverse()
        fitnesses, agents, species = zip(*zipped)
        fitnesses, agents, species = list(fitnesses), list(agents), list(species)
        representatives = {}
        for x in range(population):
            match = None
            genome = agents[x].getGenome()
            for representative in representatives.keys():
                correspondace = IoU(genome, representatives[representative])
                #print(correspondace)
                if correspondace > compatabilityThreashold:
                    match = representative
            if not match is None:
                species[x] = match
            else: # if no match  
                if species[x] in representatives.keys(): 
                    species[x] = names.get_full_name(gender='female')
                representatives[species[x]] = genome

        print('The number of species is :' + str(len(representatives.keys())))

        speciesFitnesses = representatives.copy()
        for key in speciesFitnesses.keys(): speciesFitnesses[key] = 0
        for x in range(population): speciesFitnesses[species[x]] += fitnesses[x]
        for key in speciesFitnesses.keys(): speciesFitnesses[key] = speciesFitnesses[key]/ species.count(key) 

        meanFtiness = np.mean(np.array((list(speciesFitnesses.values()))))
        adjustedSpeciesFitnesses = speciesFitnesses.copy()
        for key in adjustedSpeciesFitnesses.keys(): adjustedSpeciesFitnesses[key] = adjustedSpeciesFitnesses[key] - meanFtiness
        totalFitnesses = np.sum(np.array((list(speciesFitnesses.values()))))
        for key in adjustedSpeciesFitnesses.keys(): adjustedSpeciesFitnesses[key] = adjustedSpeciesFitnesses[key] / totalFitnesses

        for x in range(int(population*populationOverwriteRate)):
            # Pick species
            speciesToOverwrite = r.choice(list(adjustedSpeciesFitnesses.keys()))
            speciesToPropegate = r.choices(list(adjustedSpeciesFitnesses.keys()), weights=list(adjustedSpeciesFitnesses.values()))[0]

            # Pick worst agent to overwrite and random agent to propegate
            agentToOverwriteIndex = [i for i, k in enumerate(species) if k == speciesToOverwrite][-1]
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
            for index in speciesIndexs[topXtoSave:]:
                agents[index].mutate(inno)

        fitnessHistory.append(meanFtiness)
        speciesNumberHistory.append(len(representatives.keys()))

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

    agents[0].VisualizeModel()
    # save the plot as a file
    #fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
    #        format='jpeg',
    #        dpi=100,
    #        bbox_inches='tight')