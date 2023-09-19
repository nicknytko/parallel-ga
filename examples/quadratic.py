import numpy as np
import matplotlib.pyplot as plt
import sys

# Test of the genetic algorithm
# Finding a regression for a noisy quadratic function

sys.path.append('../')
import parga

np.random.seed(0)
N = 300
x = np.linspace(-1, 1, N)
y = x ** 2 + np.random.randn(N)*0.01

N_population = 100
N_coeffs = 3

def fitness(generation, weight, idx):
    y_eval = np.polyval(weight, x)
    return 1. / np.sum((y - y_eval)**2)

num_workers = 4
with parga.WorkerPool(num_workers) as pool:
    ga = parga.ParallelGA(worker_pool=pool,
                          initial_population=np.random.randn(N_population, N_coeffs),
                          fitness_func=fitness,
                          mutation_probability=0.5,
                          mutation_min_perturb=-5.,
                          mutation_max_perturb=5.)

    while True:
        ga.iteration()

        best, fit, _ = ga.best_solution()
        print(f'Generation {ga.num_generation: 3}, loss: {fit:.3e}')
        coeffs = best
        if fit > 10:
            break

    res = y - np.polyval(coeffs, x)
    var = y - np.average(y)
    det = 1 - (res@res)/(var@var)

    plt.plot(x, y, label='Noisy data')
    plt.plot(x, np.polyval(coeffs, x), 'o-', markersize=1.5, label=f'Quadratic fit, $r^2$={det:.4f}')
    plt.legend()
    plt.show(block=True)
