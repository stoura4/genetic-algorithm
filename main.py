import numpy
import GeneticAlgorithm

"""
y = target id to maximize this equation
y= w1*x1+w2*x2+w3*x3+w4*x4+w5*x5+w6*x6

"""
# import the input equation

equation_inputs= [4,-2,7,5,-11,-4.7]

#number of the weights

num_weights= len(equation_inputs)

# taille de la population

sol_per_pop= 8

# Mating pool size == number of parents

num_parents_mating= 3

# Definir la taille de la population

pop_size = (sol_per_pop,num_weights)

# La creation de la premiere generation (polupation initiale)

new_population = numpy.random.uniform (low = -4.0 , high = 4.0 , size=pop_size)

best_outputs = []
num_generations=1000

for generation in range (num_generations):
    
    print("generation:", generation)
    #evaluer la fitness de chaque chromosome = solution = regression lineaire 
    fitness = GeneticAlgorithm.cal_pop_fitness(equation_inputs , new_population)
    print("Fitness:",fitness)
    
    best_outputs.append(numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))
# The best result in the current iteration.
    print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))
    #Selectionner les meilleurs parents
    parents = GeneticAlgorithm.select_mating_pool (new_population , fitness, num_parents_mating)
    
    #generer la nouvelle generation en utilisant le croisement 
    
    offspring_crossover = GeneticAlgorithm.crossover(parents, offspring_size=(pop_size[0]-parents.shape[0],num_weights))
    print("Crossover:", offspring_crossover)
    
    #generer la nouvelle population en appliquantla mutation
    
    offspring_mutation = GeneticAlgorithm.mutation(offspring_crossover, num_mutations = 1)
    print("Mutation:", offspring_mutation)   
    
    #La creation de la nouvelle population
    
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
    
    
# evaluation de la derniere generation
fitness = GeneticAlgorithm.cal_pop_fitness (equation_inputs , new_population)

#afficher la meilleure solution
best_match_idx= numpy.where(fitness == numpy.max(fitness))

print("Best solution :" , new_population[best_match_idx, :] )
print("Best solution fitness :" , fitness[best_match_idx] )

