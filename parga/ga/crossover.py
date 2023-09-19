import numpy as np
from parga.parallel.worker import *


def greedy_selection_crossover(self):
    best, fitness, _ = self.best_solution()

    self.population[:] = best
    self.population_fitness[:] = fitness
    self.population_computed_fitness[:] = False


def roulette_selection_crossover(self):
    N_per_worker = int(np.ceil(self.population_size / self.num_workers / 2)) * 2
    chromosomes = self.population.shape[1]
    N = N_per_worker * self.num_workers

    for worker in self.workers:
        worker.send_command(WorkerCommand.create(WorkerCommand.CROSSOVER,
                                                 population=self.population,
                                                 fitness=self.population_fitness,
                                                 crossover_probability=self.crossover_probability,
                                                 selection_uniform_probability=False,
                                                 num_to_create=N_per_worker,
                                                 top_population_to_use=-1))

    self.population = np.zeros((N, chromosomes))
    self.population_fitness = np.zeros(N)
    self.population_computed_fitness = np.zeros(N, bool)

    # We will compute more pairs than needed, then discard a random subset
    indices_to_use = np.random.choice(np.arange(0, N), size=self.population_size, replace=False)

    data = self.workers.receive_all()
    for i, datum in enumerate(data):
        self.population[i * N_per_worker:(i+1) * N_per_worker] = datum['pairs']
        self.population_computed_fitness[i * N_per_worker:(i+1) * N_per_worker] = datum['pairs_computed_fitness']
        self.population_fitness[i * N_per_worker:(i+1) * N_per_worker] = datum['pairs_fitness']

    # Discard extras
    self.population = self.population[indices_to_use]
    self.population_computed_fitness = self.population_computed_fitness[indices_to_use]
    self.population_fitness = self.population_fitness[indices_to_use]


def steady_state_selection_crossover(self):
    N_to_replace = int(self.steady_state_bottom_discard * self.population_size)
    N_top_to_use = int(self.steady_state_top_use * self.population_size)
    N_per_worker = int(np.ceil(N_to_replace / self.num_workers / 2)) * 2
    chromosomes = self.population.shape[1]
    N = N_per_worker * self.num_workers

    for worker in self.workers:
        worker.send_command(WorkerCommand.create(WorkerCommand.CROSSOVER,
                                                 population=self.population,
                                                 fitness=self.population_fitness,
                                                 folds=self.model_folds,
                                                 crossover_probability=self.crossover_probability,
                                                 selection_uniform_probability=True,
                                                 num_to_create=N_per_worker,
                                                 top_population_to_use=N_top_to_use))

    # Pick worst fit individuals to replace
    indices_to_replace = np.argsort(self.population_fitness)[:N_to_replace]

    # We will compute more pairs than needed, then discard a random subset
    indices_to_use = np.random.choice(np.arange(0, N), size=N_to_replace, replace=False)

    # Receive new pairs from workers
    data = self.workers.receive_all()
    received_population = np.zeros((N, chromosomes))
    for i, datum in enumerate(data):
        received_population[i * N_per_worker:(i+1) * N_per_worker] = datum['pairs']

    # Replace subset of population
    self.population[indices_to_replace] = received_population[indices_to_use]
    self.population_computed_fitness[indices_to_replace] = False
