###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
import random

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


#function that starts population with random values between -1, 1 for the weights
def initialize_population(population_size, n_vars):
    return random.uniform(-1, 1), (population_size, n_vars)

def crossover(population, crossover_rate):
    offspring = []
    for i in range(0, population.shape[0], 2):
        if random.uniform(0, 1) < crossover_rate:
            alpha = random.uniform(0.4, 0.6)
            parent1, parent2 = population[i], population[i + 1]
            
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
            
            offspring.append(child1)
            offspring.append(child2)
            
        else:
            offspring.append(population[i])
            offspring.append(population[i+1])
                    
def tournament_selection(population, fitness, k):
    selected = []
    for i in range(len(population)):
      
        indices = random.uniform(0, len(population), k)

        best_individual = indices[np.argmax(fitness[indices])]
        selected.append(population[best_individual])
    return np.array(selected)
    
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.uniform(0, 1) < mutation_rate:
            individual[i] + np.random.normal(0, 1)
    return individual
    

def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'optimization_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    # start writing your own code from here
    generations = 100
    population_size = 100
    mutation_rate = 0.05
    crossover_rate = 0.2
    k = 3
    
    run_mode = "test"
    
    if run_mode == "train": #training phase
        population = initialize_population(population_size=population_size, n_vars=n_vars)
        fitness = evaluate(env, population)
        #global_best_fitness = np.argmax(fitness)
        #mean = np.mean(fitness)
        #std = np.st(fitness)
        solutions = [population, fitness]
        env.update_solutions(solutions)
        
        for i in range(generations):

            parents = tournament_selection(population=population, fitness=fitness, k=k)
            
            offspring_population = crossover(population=parents, crossover_rate=crossover_rate)
            
            next_generation = []
            
            for individual in offspring_population:
                next_generation += mutate(individual=individual, mutation_rate=mutation_rate)
            
            population = next_generation
            
            fitness = evaluate(env, population)
            solutions = [population, fitness]
            env.update_solutions(solutions=solutions)
            env.save_state()

    else:
        print("Hier komt test fase")


if __name__ == '__main__':
    main()