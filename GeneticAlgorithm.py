#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy

#evaluer la population 
def cal_pop_fitness (equation_inputs , pop):
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    return fitness


#selectionner les parents   
def select_mating_pool (pop , fitness, num_parents):
    parents = numpy.empty ((num_parents, pop.shape[1]))
    for parent_num in range (num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]  
        parents[parent_num,:]= pop[max_fitness_idx,:]
        fitness[max_fitness_idx]= -99999999999
    return parents 


#Crossover
def crossover(parents, offspring_size):
    offspring_crossover = numpy.empty(offspring_size)
    #le point de croisement 
    crossover_point = numpy.uint8(offspring_size[1]/2)
    
    for k in range (offspring_size[0]):
        #index de premier parent
        parent1_idx = k%parents.shape[0]
        #index de deuxieme parent
        parent2_idx = (k+1)%parents.shape[0]
        #the new offspring will have the first half of itsd genes takern from the first parents
        offspring_crossover[k,0:crossover_point]= parents[parent1_idx, 0:crossover_point]
        #the new offspring will have the second half of itsd genes takern from the second parents
        offspring_crossover[k,crossover_point:]= parents[parent2_idx, crossover_point:]   
    
    return offspring_crossover


def mutation(offspring_crossover, num_mutations = 1):
    mutations_counter = numpy.uint8(offspring_crossover.shape[1]/ num_mutations)
    
    for idx in range (offspring_crossover.shape[0]):
        gene_idx = mutations_counter -1
        for mutation_num in range(num_mutations):
            #the random value will be added to the gene
            random_value= numpy.random.uniform(-1.0,1.0,1)
            offspring_crossover[idx, gene_idx]= offspring_crossover[idx,gene_idx]+random_value
            gene_idx = gene_idx + mutations_counter
            
    return offspring_crossover



# In[ ]:





# In[ ]:




