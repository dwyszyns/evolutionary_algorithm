import numpy as np
from cec2017.functions import f1, f9
import matplotlib.pyplot as plt
import random


def make_plot(results):
    for idx, result in enumerate(results):
        plt.plot(
            [i for i in range(0, result.num_of_iterations + 1)],
            result.min_func_value,
            label=f"Result {idx+1}",
        )
    plt.xlabel("Iteration")
    plt.ylabel("Value of the objective function")
    plt.title("Plot of the objective function values from iterations")
    plt.legend()
    plt.show()


class optim_result:
    def __init__(self, minimum, min_func_value, num_of_iterations):
        self.minimum = minimum
        self.min_func_value = min_func_value
        self.num_of_iterations = num_of_iterations


class Params_t:
    def __init__(
        self,
        sigma,
        a,
        max_iter,
    ):
        self.sigma = sigma
        self.a = a
        self.max_iter = max_iter


def one_fifth_success_rule(successful_mutations, i, sigma, a):
    if i % a == 0:
        if successful_mutations / a > 1 / 5:
            sigma *= 1.22
        elif successful_mutations / a < 1 / 5:
            sigma *= 0.82
        successful_mutations = 0
    return sigma, successful_mutations


class one_fifth_succ_rule():
    def __init__(self, adaptation_interval):
        self.adaptation_interval = adaptation_interval
        self.target_prob = 1/5

    # def update(old_step_size, ...) -> float:
    #     return old_step_size * -- to ta funkcja one_fifth_success_rule
    

# można dodać parametr, który sprawdza średnią
# epsilon
# ile licz sprawdzamy


def mutate(reproduction, population, mutation_rate, population_size):
    return [
        population[i] + mutation_rate * reproduction[i] for i in range(population_size)
    ]
    # numpy sum i multiply


def create_population(population_size):
    return [random.gauss(0, 1) for _ in range(population_size)]





def evel_1p1(
    func, population_size, population, params: Params_t, stepsize_adaptation_policy=0
) -> optim_result:
    y_min_tab = [func(population)]
    y_min = func(population)
    min_pop = population
    successful_mutations = 0
    for i in range(params.max_iter):
        random_point = create_population(population_size)
        # reproduction = create_population(population_size)
        # new_point
        mut_population = mutate(random_point, population, params.sigma, population_size)
        # y_min, x_min = update_min_point(mut_population)
        zmienna = func(mut_population)
        if y_min >= zmienna:
            successful_mutations += 1
            y_min = func(mut_population)
            min_pop = mut_population
            population = mut_population
        # sigma = stepsize_adaptation_policy(...)
        params.sigma, successful_mutations = one_fifth_success_rule(
            successful_mutations, i, params.sigma, params.a
        )
        y_min_tab.append(y_min)
    return optim_result(min_pop, y_min_tab, params.max_iter)


def f(x):
    return x[0] * x[0]


if __name__ == "__main__":
    # population = create_population(10)
    population = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    params_t = Params_t(10.0, 20.0, 100000)
    #range max 10
    for _ in range(10):
        results = evel_1p1(
            f1, population_size=10, population=population, params=params_t
        )
        print(_, "Minimum dla f1:", results.minimum, "Wartość funkcji celu:", results.min_func_value[-1])
        # make_plot(results)
    # print("wartosc funkcji celu:", result)
#rysowaćwykres funkcji czy średnią?

#wywalic population size