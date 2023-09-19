import numpy as np
from abc import ABC, abstractmethod
from parga.parallel.worker import *


class BaseMutation(ABC):
    @abstractmethod
    def __init__(self, ga, kwargs):
        pass

    @abstractmethod
    def mutate(self, ga, best, fitness):
        pass

    @abstractmethod
    def post_mutate(self, ga, old_best, old_fitness, new_best, new_fitness):
        pass


class CustomMutationWrapper(BaseMutation):
    def __init__(self, mutation):
        self.mutation = mutation

    def mutate(self, ga, best, fitness):
        self.mutation(ga, best, fitness)


class ApproximateGradientMomentumMutation(BaseMutation):
    def __init__(self, ga, kwargs):
        self.running_gradient = np.zeros(ga.chromosome_size)
        self.gradient_avg = 5
        self.gradient_mix = 0.2

    def mutate(self, ga, best, fitness):
        new_children = np.where(ga.population_computed_fitness == False)[0]
        for i, worker in enumerate(ga.workers):
            local_indices = new_children[i::ga.num_workers]
            local_population = ga.population[local_indices]
            if len(local_indices) != 0:
                worker.send_command(WorkerCommand.create(
                    WorkerCommand.MUTATION,
                    population=local_population,
                    indices=local_indices,
                    folds=ga.model_folds,
                    mutation_probability=ga.mutation_probability,
                    mutation_perturb=(ga.mutation_min_perturb, ga.mutation_max_perturb),
                    momentum=self.running_gradient,
                    momentum_mix=self.gradient_mix))
            else:
                worker.send_command(WorkerCommand.create(WorkerCommand.NOOP))

        # Now, assemble data we get back from the workers
        data = ga.workers.receive_all()
        for datum in data:
            if datum['command'] == WorkerCommand.NOOP:
                continue
            local_indices = datum['indices']
            local_population = datum['population']
            ga.population[local_indices] = local_population
            ga.population_computed_fitness[local_indices] = False

    def post_mutate(self, ga, old_best, old_fitness, new_best, new_fitness):
        g = self.gradient_avg
        self.running_gradient = self.running_gradient * ((g-1)/g)
        if old_fitness != new_fitness:
            self.running_gradient = self.running_gradient + (new_best - old_best) * (1/g)

class SPSAMutation(BaseMutation):
    def __init__(self, ga, kwargs):
        self.approximate_gradient = np.zeros(ga.chromosome_size)

    def mutate(self, ga, best, fitness):
        new_children = np.where(self.population_computed_fitness == False)[0]
        num_new = len(new_children)

        perturbation = np.random.binomial(1, 0.5, size=(num_new, ga.chromosome_size)) * 2.0 - 1.0
        # todo...
