# from interval import Interval
import random
import copy
import numpy as np
import math
from tqdm import tqdm

from scipy.spatial.distance import cdist

class Individual:
    def __init__(self):
        self.chromosome = None
        self.fitness = None
        self.rank = None
        self.crowding_distance = None

    def __lt__(self, other):
        return self.rank < other.rank or (self.rank == other.rank and self.crowding_distance > other.crowding_distance)

class nsgaii:
    def __init__(self,dv,eva_fun,num_generations,tournament_size,crossover_rate,mutation_rate,pop_size):
        self.dv=dv
        self.eva_fun=eva_fun
        self.num_generations=num_generations
        self.tournament_size=tournament_size
        self.crossover_rate=crossover_rate
        self.mutation_rate=mutation_rate
        self.population=None
        self.pop_size=pop_size


    def init_pop(self):
        self.population = []
        for i in range(len(self.dv)):
            individual=Individual()
            individual.chromosome=self.dv[i]
            self.population.append(individual)

    def run(self,pop=None):
        if pop==None:
            self.init_pop()
        else:
            self.population=pop

        # for i in tqdm(range(self.num_generations)):
        for i in range(self.num_generations):
            # print(f"第{i+1}次迭代")
            # Evaluate fitness and assign rank and crowding distance to individuals
            self.evaluate_population(self.population)

            # Create offspring population through tournament selection, crossover and mutation
            offspring_population = []
            while len(offspring_population) < self.pop_size*2:
                parent1 = self.tournament_selection(self.population, self.tournament_size)
                parent2 = self.tournament_selection(self.population, self.tournament_size)

                offspring1, offspring2 =self.crossover(parent1, parent2, self.crossover_rate)
                if offspring1 is not None:
                    # print(parent1.chromosome,parent2.chromosome,offspring1.chromosome,offspring2.chromosome)
                    offspring1 = self.mutation(offspring1, self.mutation_rate)
                    offspring2 = self.mutation(offspring2,self.mutation_rate)
                    offspring_population.append(offspring1)
                    offspring_population.append(offspring2)
                    # print(parent1.chromosome)

            # Combine parent and offspring populations and perform non-dominated sorting and crowding distance calculation

            combined_population = self.population + offspring_population

            fronts = self.non_dominated_sort(combined_population)

            for front in fronts:
                if len(front)!=0:
                    self.calculate_crowding_distance(front)
            # Select the next generation population based on non-dominated sorting and crowding distance
            self.population = []
            for front in fronts:
                for i in front:
                    self.population.append(i)
                    if len(self.population) >= self.pop_size:
                        break
            # print(self.cal_HV())
            # print(len(self.population))
        return self.population

    def evaluate_individual(self,individual):
        # Evaluate the fitness of an individual
        # Update the individual's fitness attribute
        return([self.eva_fun.cal_spi(individual.chromosome),self.eva_fun.cal_cost(individual.chromosome),self.eva_fun.cal_time(individual.chromosome)])

    def evaluate_population(self,population):
        # Evaluate the fitness of all individuals in a population
        # Update the rank and crowding distance attributes of each individual
        for individual in population:
            individual.fitness = self.evaluate_individual(individual)

    def tournament_selection(self, population, tournament_size):
        # Select an individual from a population using tournament selection
        # Return the selected individual
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1, parent2, crossover_prob):
        # Create a copy of the parents to avoid modifying the original individuals
        # Check if crossover should be applied
        parent1_copy = copy.deepcopy(parent1)
        parent2_copy = copy.deepcopy(parent2)
        path1=parent1.chromosome[::2]
        path2=parent2.chromosome[::2]
        st1=parent1.chromosome[1::2]
        st2=parent2.chromosome[1::2]
        path1_copy = copy.deepcopy(path1)
        path2_copy = copy.deepcopy(path2)
        st1_copy = copy.deepcopy(st1)
        st2_copy = copy.deepcopy(st2)
        positions = []
        def qcpath(path,st):
            dic = {}
            for i in range(len(path)):
                dic[path[i]] = st[i]
            return(list(dic.keys()),list(dic.values()))

        # 遍历list1中的元素
        for i, x in enumerate(path1):
            # 如果元素也在list2中出现，则将它们的位置保存到positions列表中
            if x in path2:
                j = path2.index(x)
                positions.append((i, j))
        if len(positions)==0:
            return None,None
        if path1 == path2 and random.random() < crossover_prob:
            crossover_point = random.randint(0, len(path1) - 1)
            st1_copy[crossover_point:] = st2[crossover_point:]
            st2_copy[crossover_point:] = st2[crossover_point:]
            if len(st1_copy)!=len(st1):
                path1_copy,st1_copy=qcpath(path1_copy,st1_copy)
                path2_copy, st2_copy = qcpath(path2_copy, st2_copy)
            parent1_copy.chromosome = [list(zip(path1_copy, st1_copy))[i][j] for i in range(len(path1_copy)) for j in range(len(list(zip(path1_copy, st1_copy))[0]))]
            parent2_copy.chromosome = [list(zip(path2_copy, st2_copy))[i][j] for i in range(len(path2_copy)) for j in range(len(list(zip(path2_copy, st2_copy))[0]))]
            return parent1_copy, parent2_copy

        if path1!=path2 and random.random() < crossover_prob:
            # Select a random crossover point
            crossover_point = random.randint(0, len(positions) - 1)
            crossover_point1=positions[crossover_point][0]
            crossover_point2 = positions[crossover_point][1]
            path1_copy[crossover_point1:]=path2[crossover_point2:]
            st1_copy[crossover_point1:] = st2[crossover_point2:]
            path2_copy[crossover_point2:]= path1[crossover_point1:]
            st2_copy[crossover_point2:] = st1[crossover_point1:]
            # Swap the chromosomes of the parents from the crossover point onwards
            if len(st1_copy)!=len(st1):
                path1_copy,st1_copy=qcpath(path1_copy,st1_copy)
                path2_copy,st2_copy = qcpath(path2_copy,st2_copy)

            parent1_copy.chromosome = [list(zip(path1_copy, st1_copy))[i][j] for i in range(len(path1_copy)) for j in range(len(list(zip(path1_copy, st1_copy))[0]))]
            parent2_copy.chromosome = [list(zip(path2_copy, st2_copy))[i][j] for i in range(len(path2_copy)) for j in range(len(list(zip(path2_copy, st2_copy))[0]))]
        return parent1_copy, parent2_copy

    def mutation(self,individual, mutation_rate):
        # Perform mutation on an individual with a given mutation rate
        # Return the mutated individual
        # Perform mutation with a given probability
        # Perform crossover between two parents with a certain probability
        # Return two offspring
        if random.random() < mutation_rate:
            # Select a random gene to mutate
            while True:
                gene_index = random.randint(0, len(individual.chromosome) - 1)
                if gene_index %2 == 1:
                    break

            # Mutate the gene with a small random value
            individual.chromosome[gene_index] = random.randint(-1, 4)

        return individual

    def non_dominated_sort(self,population):
        # Perform non-dominated sorting on a population
        # Return a list of fronts, where each front is a list of individuals
        def dominates(x, y):
            # return all(nx<=ny) and any(nx<ny)
            return all(x.fitness[i]<=y.fitness[i] for i in range(len(x.fitness))) and \
                   any(x.fitness[i]<y.fitness[i] for i in range(len(x.fitness)))

        domination_count = [0] * len(population)
        domination_set = [[] for _ in range(len(population))]

        for i in range(len(population)):
            for j in range(len(population)):
                if i == j:
                    continue
                if dominates(population[i], population[j]):
                    domination_set[i].append(j)
                elif dominates(population[j], population[i]):
                    domination_count[i] += 1

        fronts = [[]]
        for i in range(len(population)):
            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in domination_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)
        for i in range(len(fronts)):
            for j in range(len(population)):
                if j in fronts[i]:
                    population[j].rank=i
        pop_front=[]
        for i in range(len(fronts)):
            tmp=[]
            for j in range(len(fronts[i])):
                tmp.append(population[fronts[i][j]])
            pop_front.append(tmp)

        return pop_front

    def calculate_crowding_distance(self,front):
        # Calculate the crowding distance of individuals in a front
        # Update the crowding distance attribute of each individual
        num_objectives = len(front[0].fitness)
        num_individuals = len(front)
        # Initialize crowding distance of all individuals to zero
        for individual in front:
            individual.crowding_distance = 0

        # Calculate crowding distance for each objective
        for i in range(num_objectives):
            # Sort individuals by fitness for objective i
            front.sort(key=lambda individual: individual.fitness[i])
            # Assign infinite crowding distance to boundary individuals
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            # Calculate crowding distance for inner individuals
            dmax= front[-1].fitness[i]
            dmin= front[0].fitness[i]
            for j in range(1, num_individuals - 1):
                front[j].crowding_distance += (front[j + 1].fitness[i] - front[j - 1].fitness[i]) / (
                            dmax - dmin+0.00001)

        front.sort(key=lambda individual: -individual.crowding_distance)




