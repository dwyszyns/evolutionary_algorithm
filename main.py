import numpy as np
from cec2017.functions import f1, f9
import matplotlib.pyplot as plt
import random


def make_normal_plot(results):
    for i in range(len(results)):
        all_func_values = np.array(results[i][1:])
        mean_func_values = np.mean(all_func_values, axis=0)
        iterations = np.arange(len(results[i][0]))
        plt.plot(iterations, mean_func_values, label=f"Result {i+1}")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Value of the function")
    plt.legend()
    plt.title("Plot of the function values from iterations")
    plt.show()


def make_ribbon_plot(results):
    _, ax = plt.subplots()
    for i in range(len(results)):
        all_func_values = np.array(results[i][1:])
        min_func_values = np.min(all_func_values, axis=0)
        max_func_values = np.max(all_func_values, axis=0)
        mean_func_values = np.mean(all_func_values, axis=0)
        iterations = np.arange(len(results[i][0]))
        ax.fill_between(iterations, min_func_values, max_func_values, alpha=0.2)
        ax.plot(iterations, mean_func_values, alpha=0.7, label=f"Result {i+1}")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value of the function")
    ax.set_title("Ribbon Plot of the function values from iterations")
    plt.legend()
    plt.show()


class simulation_results:
    def __init__(self, length):
        self.optim_results = [[] for _ in range(length)]

    def add_new_result(self, optim_result, i):
        self.optim_results[i].append(optim_result)


class optim_result:
    def __init__(self, min_point, min_func_value, num_of_iterations=1, func_values=[]):
        self.min_point = min_point
        self.min_func_value = min_func_value
        self.num_of_iterations = num_of_iterations
        self.func_values = func_values


class params_t:
    def __init__(self, mutation_stren, max_iter):
        self.mutation_stren = mutation_stren
        self.max_iter = max_iter


class one_fifth_succ_rule:
    def __init__(self, adapt_interval):
        self.adapt_interval = adapt_interval
        self.target_prob = 1 / 5

    def update_mutation_stren(self, successful_mutations, i, mutation_stren):
        if i % self.adapt_interval == 0:
            if successful_mutations / self.adapt_interval > self.target_prob:
                mutation_stren *= 1.22
            elif successful_mutations / self.adapt_interval < self.target_prob:
                mutation_stren *= 0.82
            successful_mutations = 0
        return successful_mutations, mutation_stren


class Evolution_Solver:
    def __init__(self, x0, params, mut_stren_adapt_policy):
        self.params = params
        self.success_mutations = 0
        self.result = optim_result(x0, 0, params.max_iter, [])
        self.point_size = len(x0)
        self.mut_stren_adapt_policy = mut_stren_adapt_policy

    def mutate(self, mut_point, mutation_stren):
        return [
            self.result.min_point[i] + mutation_stren * mut_point[i]
            for i in range(self.point_size)
        ]

    def update_min_point(self, f_mut_pint, mut_point):
        if self.result.min_func_value >= f_mut_pint:
            self.success_mutations += 1
            self.result.min_func_value = f_mut_pint
            self.result.min_point = mut_point
        self.result.func_values.append(self.result.min_func_value)

    def solve(self, f):
        self.result.min_func_value = f(self.result.min_point)
        self.result.func_values.append(self.result.min_func_value)
        for i in range(self.params.max_iter):
            random_point = [random.gauss(0, 1) for _ in range(self.point_size)]
            mut_point = self.mutate(random_point, self.params.mutation_stren)
            self.update_min_point(f(mut_point), mut_point)
            (
                self.success_mutations,
                self.params.mutation_stren,
            ) = self.mut_stren_adapt_policy.update_mutation_stren(
                self.success_mutations, i, self.params.mutation_stren
            )


if __name__ == "__main__":
    x0 = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    mut_stren_adapt_policy = one_fifth_succ_rule(10.0)
    params_list = [
        params_t(0.0001, 10000),
        params_t(0.1, 10000),
        params_t(1, 10000),
        params_t(1000, 10000),
    ]
    results = simulation_results(len(params_list))
    for i in range(len(params_list)):
        for _ in range(10):
            evolution_solver = Evolution_Solver(
                x0, params_list[0], mut_stren_adapt_policy
            )
            evolution_solver.solve(f1)
            results.add_new_result(evolution_solver.result.func_values, i)
    make_ribbon_plot(results.optim_results)
