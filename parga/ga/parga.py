import numpy as np
from parga.parallel.worker import *
import parga.ga.crossover as crossover
import parga.ga.mutation as mutation
import types


class ParallelGA:
    '''
    Parallel implementation of Genetic Algorithm, used to train
    neural networks without needing any gradient information
    '''

    def __init__(self, **kwargs):
        '''
        Initializes the GA method

        Keyword Arguments
        ----------
        initial_population : np.ndarray
          population x chromosomes length array, representing the initial population
        fitness_func : callable, (individual, idx) -> fitness
          Function to be called to evaluate network fitness.  Must be pickle-able.
        crossover_probability : float (default 0.5)
          Probability in [0, 1] that offspring will be a random crossover between two parents
        mutation_probability : float (default 0.3)
          Probability in [0, 1] that an individual will be randomly mutated
        mutation_min_perturb : float (default -1)
          Minimum value to perturb a chromosome when mutating
        mutation_max_perturb : float (default 1)
          Maximum value to perturb a chrosome when mutating
        steady_state_top_use : float (default 1/3)
          Percent of the population to use for breeding when steady state selection is used
        steady_state_bottom_discard : float (default 1/3)
          Percent of the population to discard when steady state selection is used
        selection : str {steady_state, roulette, greedy}
          Selection method to use
        num_workers : int (default 2)
          Number of worker processes to use
        '''

        self.population = np.copy(kwargs.get('initial_population', np.zeros((1,1))))
        self.population_size = self.population.shape[0]
        self.chromosome_size = self.population.shape[1]
        self.population_fitness = np.zeros(self.population_size)
        self.population_computed_fitness = np.zeros(self.population_size, bool)

        self.fitness_func = kwargs.get('fitness_func', None)
        self.crossover_probability = kwargs.get('crossover_probability', 0.5)
        self.mutation_probability = kwargs.get('mutation_probability', 0.3)

        self.mutation_min_perturb = kwargs.get('mutation_min_perturb', -1.)
        self.mutation_max_perturb = kwargs.get('mutation_max_perturb',  1.)

        self.steady_state_top_use = kwargs.get('steady_state_top_use', 1./3.)
        self.steady_state_bottom_discard = kwargs.get('steady_state_bottom_discard', 1./3.)

        self.model_folds = kwargs.get('model_folds', None)
        self.restart_iteration = kwargs.get('restart_every', None)

        self.num_generation = 0

        # Selection/crossover
        self.selection_to_use = kwargs.get('selection', 'steady_state')
        if callable(self.selection_to_use):
            self.selection_method = lambda: self.selection_to_use(self)
        else:
            if not self.selection_to_use in ['steady_state', 'roulette', 'greedy']:
                raise RuntimeError(f'Unknown selection method: {self.selection_to_use}')
            self.selection_method = types.MethodType({
                'steady_state': crossover.steady_state_selection_crossover,
                'roulette': crossover.roulette_selection_crossover,
                'greedy': crossover.greedy_selection_crossover,
            }[self.selection_to_use], self)

        # Mutation
        mtn = kwargs.get('mutation', None)
        if callable(mtn):
            self.mutation = mutation.CustomMutationWrapper(mtn)
        elif isinstance(mtn, mutation.BaseMutation):
            self.mutation = mtn
        else:
            if (mtn is None or
                mtn == 'default' or
                mtn == 'momentum'):
                self.mutation = mutation.ApproximateGradientMomentumMutation(self, kwargs)
            elif mtn == 'spsa':
                self.mutation = mutation.SPSAMutation(self, kwargs)
            else:
                raise RuntimeError(f'Unknown mutation method: {mtn}')

        self.workers = kwargs.get('worker_pool')
        self.num_workers = len(self.workers)


    def compute_fitness(self):
        if np.all(self.population_computed_fitness):
            return

        # Only compute fitness for population where it is unknown
        to_compute = np.where(~self.population_computed_fitness)[0]

        # Divvy up the population for the workers using a cyclic mapping.
        # The mapping we use doesn't really matter here, but I'm lazy and it's
        # really easy to slice a cycling mapping in Python.
        for i, worker in enumerate(self.workers):
            local_indices = to_compute[i::self.num_workers]
            local_population = self.population[local_indices]
            if len(local_indices) != 0:
                worker.send_command(WorkerCommand.create(WorkerCommand.FITNESS,
                                                         generation=self.num_generation,
                                                         population=local_population,
                                                         indices=local_indices,
                                                         fitness_func=self.fitness_func))
            else:
                worker.send_command(WorkerCommand.create(WorkerCommand.NOOP))

        # Now, assemble data we get back from the workers
        data = self.workers.receive_all()
        for datum in data:
            if datum['command'] == WorkerCommand.NOOP:
                continue
            local_indices = datum['indices']
            local_fitness = datum['fitness']
            self.population_fitness[local_indices] = local_fitness
            self.population_computed_fitness[local_indices] = True


    def restart(self):
        best, fitness, _ = self.best_solution()
        self.population_computed_fitness[:] = False

        # Copy best solution to all population, then we will mutate all but first
        self.population[:] = best
        self.population_computed_fitness[0] = True
        self.population_fitness[0] = fitness

        rand = np.random.RandomState()
        self.population[1:] += rand.uniform(low=-1.0, high=1.0, size=(self.population_size-1, self.population.shape[1]))


    def iteration(self):
        '''
        Performs one iteration of the GA
        '''

        if (self.restart_iteration is not None and
            self.num_generation > 0 and
            self.num_generation % self.restart_iteration == 0):
            self.restart()

        self.num_generation += 1
        best, fitness, _ = self.best_solution()
        self.selection_method()
        if self.mutation_probability != 0.0:
            self.mutation.mutate(self, best, fitness)
        self.compute_fitness()

        # replace worst with previous best, so we never totally remove the best solution we have
        # this gives us a monotonically increasing fitness
        worst = np.argmin(self.population_fitness)
        self.population[worst] = best
        self.population_fitness[worst] = fitness

        new_best, new_fitness, _ = self.best_solution()
        self.mutation.post_mutate(self, best, fitness, new_best, new_fitness)


    def stochastic_iteration(self):
        self.num_generation += 1
        # Recompute every network in population, so that local fitness
        # is relative to computed minibatch
        self.population_computed_fitness[:] = False
        self.compute_fitness()

        best, fitness, _ = self.best_solution()

        self.selection_method()
        self.mutation()
        self.compute_fitness()

        # Replace worst with previous best
        worst = np.argmin(self.population_fitness)
        self.population[worst] = best
        self.population_fitness[worst] = fitness


    def best_solution(self):
        '''
        Return the best solution that has been encountered so far

        Returns
        -------
        (individual, fitness, index)
        '''

        self.compute_fitness()
        idx = np.argmax(self.population_fitness)
        return self.population[idx].copy(), self.population_fitness[idx], idx


    def parallel_map(self, iterable, function, extra_args):
        return self.workers.map(iterable, function, extra_args)
