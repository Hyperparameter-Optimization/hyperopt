import numpy as np


class Gene:
    def __init__(self, parameter_info, iterations):
        self.parameter_info = parameter_info
        self.iterations = iterations
        if bool(self.parameter_info['exp']):
            self.sigma_initial = (
                np.exp(parameter_info['max']) - np.exp(parameter_info['min'])) / 4
        else:
            self.sigma_initial = (parameter_info['max'] - parameter_info['min']) / 4
        self.sigma_step = float(self.sigma_initial) / self.iterations

    def create_gene(self, key, value, parameter_info):
        self.value = value
        self.key = key

    def initialize_gene(self):
        if bool(self.parameter_info['int']):
            self.value = np.random.randint(
                low=self.parameter_info['min'],
                high=self.parameter_info['max']
            )
        else:
            self.value = np.random.uniform(
                self.parameter_info['min'],
                self.parameter_info['max']
            )
        if bool(self.parameter_info['exp']):
            self.value = np.exp(self.value)
        self.key = self.parameter_info['parameter']

    def mutate(self):
        self.sigma = self.sigma_initial - (self.sigma_step * self.iteration)
        if np.random.uniform() > 0.5:
            self.value += np.random.normal(scale=self.sigma)
        if bool(self.parameter_info['exp']):
            if self.value > np.exp(self.parameter_info['max']):
                self.value = np.exp(self.parameter_info['max'])
            elif self.value < np.exp(self.parameter_info['min']):
                self.value = np.exp(self.parameter_info['min'])
        else:
            if self.value > self.parameter_info['max']:
                self.value = self.parameter_info['max']
            elif self.value < self.parameter_info['min']:
                self.value = self.parameter_info['min']

    def set_iteration(self, iteration):
        self.iteration = iteration


class Chromosome:
    def __init__(self, parameter_infos, settings):
        self.iteration = 0
        self.parameter_infos = parameter_infos
        self.settings = settings

    def initialize_chromosome(self):
        self.parameters = {}
        self.genes = []
        for parameter_info in self.parameter_infos:
            gene = Gene(parameter_info, self.settings['iterations'])
            gene.initialize_gene()
            self.parameters[gene.key] = gene.value
            self.genes.append(gene)

    def mutation(self, iteration):
        self.parameters = {}
        for gene in self.genes:
            gene.set_iteration(iteration)
            if self.settings['mutation_chance'] > np.random.uniform():
                gene.mutate()
            self.parameters[gene.key] = gene.value

    def set_iteration(self, iteration):
        self.iteration = iteration

    def create_chromosome(self, parameters, iteration):
        self.parameters = parameters
        self.genes = []
        for parameter_info in self.parameter_infos:
            key = parameter_info['parameter']
            gene = Gene(parameter_info, self.settings['iterations'])
            gene.create_gene(
                key,
                parameters[key],
                parameter_info
            )
            self.genes.append(gene)

    def set_fitness(self, fitness):
        self.fitness = fitness


class Subpopulation:
    def __init__(self, parameter_infos, fitness_function, settings):
        self.iteration = 0
        self.parameter_infos = parameter_infos
        self.settings = settings
        self.iterations = self.settings['iterations']
        self.members = self.initialize_members(self.settings['sample_size'])
        self.fitness_function = fitness_function

    def initialize_members(self, sample_size):
        population = []
        for i in range(sample_size):
            chromosome = Chromosome(self.parameter_infos, self.settings)
            chromosome.initialize_chromosome()
            population.append(chromosome)
        return population

    def evaluate_chromosomes(self): 
        chromosome_dicts = []
        for chromosome in self.members:
            chromosome_dict = {
                gene.key: gene.value for gene in chromosome.genes
            }
            chromosome_dicts.append(chromosome_dict)
        fitnesses = self.fitness_function(chromosome_dicts, self.settings)
        for fitness, chromosome in zip(fitnesses, self.members):
            chromosome.set_fitness(fitness)

    def crossover(self):
        keys = self.members[0].parameters.keys()
        self.members = []
        for pair in self.parent_pairs:
            offspring = {}
            for key in keys:
                if np.random.uniform() > 0.5:
                    offspring[key] = pair[0].parameters[key]
                else:
                    offspring[key] = pair[1].parameters[key]
            chromosome = Chromosome(self.parameter_infos, self.settings)
            chromosome.create_chromosome(offspring, self.iteration)
            self.members.append(chromosome)

    def selection(self):
        self.parent_pairs = []
        nr_parent_pairs = self.settings['sample_size'] - (
            self.settings['nr_elites'] + self.settings['nr_culled']
        )
        for i in range(nr_parent_pairs):
            parent_pair = self.tournament()
            self.parent_pairs.append(parent_pair)

    def tournament(self, winner_probability=0.4, tournament_size=5):
        parent_pair = []
        random_idx = np.random.choice(
            len(self.members), size=tournament_size, replace=False
        )
        competing_chromosomes = np.array(self.members)[random_idx]
        fitnesses = [chromosome.fitness for chromosome in competing_chromosomes]
        while len(parent_pair) < 2:
            best_chromosome_idx = np.argmax(fitnesses)
            best_chromosome = competing_chromosomes[best_chromosome_idx]
            if np.random.uniform() < winner_probability:
                parent_pair.append(best_chromosome)
            competing_chromosomes = list(competing_chromosomes)
            competing_chromosomes.remove(best_chromosome)
            fitnesses.remove(fitnesses[best_chromosome_idx])
            if len(parent_pair) == 0 and len(competing_chromosomes) == 2:
                parent_pair = competing_chromosomes
            if len(parent_pair) < 2 and len(competing_chromosomes) == 1:
                parent_pair.append(competing_chromosomes[0])
        return parent_pair

    def culling(self):
        for i in range(self.settings['nr_culled']):
            scores = [chromosome.fitness for chromosome in self.members]
            idx = np.argmin(scores)
            self.members.remove(self.members[idx])

    def elitism(self):
        self.elites = []
        removable_pop = list(self.members)
        for i in range(self.settings['nr_elites']):
            scores = [chromosome.fitness for chromosome in removable_pop]
            idx = np.argmax(scores)
            self.elites.append(removable_pop[idx])
            removable_pop.remove(removable_pop[idx])

    def mutate(self):
        for chromosome in self.members:
            chromosome.mutation(self.iteration)

    def survivalOfTheFittest(self):
        self.evaluate_chromosomes()
        for i in range(self.settings['iterations']):
            self.iteration = i
            self.elitism()
            self.culling()
            self.selection()
            self.crossover()
            self.mutate()
            new_members = self.initialize_members(self.settings['nr_culled'])
            self.members.extend(self.elites)
            self.members.extend(new_members)
            self.evaluate_chromosomes()
        return self.members


class Population:
    def __init__(self, parameter_infos, fitness_function, settings):
        self.iteration = 0
        self.parameter_infos = parameter_infos
        self.settings = settings
        self.iterations = self.settings['iterations']
        self.fitness_function = fitness_function
        self.initialize_subpopulations()

    def initialize_population(self, sample_size):
        population = []
        for i in range(sample_size):
            chromosome = Chromosome(self.parameter_infos, self.settings)
            chromosome.initialize_chromosome()
            population.append(chromosome)
        return population

    def initialize_subpopulations(self):
        self.subpopulations = []
        subpop_size = int(
            self.settings['sample_size'] / self.settings['nr_subpop'])
        subpop_iterations = int(
            self.settings['iterations'] * self.settings['subpop_iter_frac']
        )
        subpop_settings = self.settings.copy()
        subpop_settings['iterations'] = subpop_iterations
        subpop_settings['sample_size'] = subpop_size
        subpop_settings['nr_culled'] = int(
            self.settings['nr_culled'] / self.settings['nr_subpop'])
        subpop_settings['nr_elites'] = int(
            self.settings['nr_elites'] / self.settings['nr_subpop'])
        for i in range(self.settings['nr_subpop']):
            subpopulation = Subpopulation(
                self.parameter_infos, self.fitness_function, subpop_settings
            )
            self.subpopulations.append(subpopulation)

    def evolve_subpopulations(self):
        self.population = []
        for subpopulation in self.subpopulations:
            evolvedSubpop = subpopulation.survivalOfTheFittest()
            self.population.extend(evolvedSubpop)

    def evaluate_chromosomes(self):
        chromosome_dicts = []
        for chromosome in self.population:
            chromosome_dict = {
                gene.key: gene.value for gene in chromosome.genes
            }
            chromosome_dicts.append(chromosome_dict)
        fitnesses = self.fitness_function(chromosome_dicts, self.settings)
        for fitness, chromosome in zip(fitnesses, self.population):
            chromosome.set_fitness(fitness)

    def crossover(self):
        keys = self.population[0].parameters.keys()
        self.population = []
        for pair in self.parent_pairs:
            offspring = {}
            for key in keys:
                if np.random.uniform() > 0.5:
                    offspring[key] = pair[0].parameters[key]
                else:
                    offspring[key] = pair[1].parameters[key]
            chromosome = Chromosome(self.parameter_infos, self.settings)
            chromosome.create_chromosome(offspring, self.iteration)
            self.population.append(chromosome)

    def selection(self):
        self.parent_pairs = []
        nr_parent_pairs = self.settings['sample_size'] - (
            self.settings['nr_elites'] + self.settings['nr_culled']
        )
        for i in range(nr_parent_pairs):
            parent_pair = self.tournament()
            self.parent_pairs.append(parent_pair)

    def tournament(self, winner_probability=0.4, tournament_size=5):
        parent_pair = []
        random_idx = np.random.choice(
            len(self.population), size=tournament_size, replace=False
        )
        competing_chromosomes = np.array(self.population)[random_idx]
        fitnesses = [chromosome.fitness for chromosome in competing_chromosomes]
        while len(parent_pair) < 2:
            best_chromosome_idx = np.argmax(fitnesses)
            best_chromosome = competing_chromosomes[best_chromosome_idx]
            if np.random.uniform() < winner_probability:
                parent_pair.append(best_chromosome)
            competing_chromosomes = list(competing_chromosomes)
            competing_chromosomes.remove(best_chromosome)
            fitnesses.remove(fitnesses[best_chromosome_idx])
            if len(parent_pair) == 0 and len(competing_chromosomes) == 2:
                parent_pair = competing_chromosomes
            if len(parent_pair) < 2 and len(competing_chromosomes) == 1:
                parent_pair.append(competing_chromosomes[0])
        return parent_pair

    def culling(self):
        for i in range(self.settings['nr_culled']):
            scores = [chromosome.fitness for chromosome in self.population]
            idx = np.argmin(scores)
            self.population.remove(self.population[idx])

    def elitism(self):
        self.elites = []
        removable_pop = list(self.population)
        for i in range(self.settings['nr_elites']):
            scores = [chromosome.fitness for chromosome in removable_pop]
            idx = np.argmax(scores)
            self.elites.append(removable_pop[idx])
            removable_pop.remove(removable_pop[idx])

    def mutate(self):
        for chromosome in self.population:
            chromosome.mutation(self.iteration)

    def survivalOfTheFittest(self):
        self.evolve_subpopulations()
        print('SUBPOPULATIONS EVOLVED')
        self.evaluate_chromosomes()
        subpop_iter = int(
            self.settings['subpop_iter_frac'] * self.settings['iterations'])
        for i in range(subpop_iter, self.settings['iterations']):
            self.iteration = i
            self.elitism()
            self.culling()
            self.selection()
            self.crossover()
            self.mutate()
            new_members = self.initialize_population(self.settings['nr_culled'])
            self.population.extend(self.elites)
            self.population.extend(new_members)
            self.evaluate_chromosomes()
        fitnesses = [chromosome.fitness for chromosome in self.population]
        idx = np.argmax(fitnesses)
        print('Best fitness: ' + str(fitnesses[idx]))
        print('Best parameters: ' + str(self.population[idx].parameters))
        return self.population[idx].parameters, fitnesses[idx]
