import numpy as np
from cec2017.functions import f1, f9
import matplotlib.pyplot as plt
import random


def make_plot(results):
    labels = ['adapt_interval = 0.005', 'adapt_interval = 1', 'adapt_interval = 50', 'adapt_interval = 1000']
    for i in range(len(results)):
        all_func_values = np.array(results[i][1:])
        mean_func_values = np.mean(all_func_values, axis=0)
        iterations = np.arange(len(results[i][0]))
        plt.plot(iterations, mean_func_values)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Value of the function")
    plt.title("Plot of the function values from iterations")
    plt.legend()
    plt.savefig("wykres_norm_f9_max_iter.png")
    plt.show()  


def make_ribbon_plot(results):
    labels =['adapt_interval = 0.005', 'adapt_interval = 1', 'adapt_interval = 50', 'adapt_interval = 1000']
    _, ax = plt.subplots()
    for i in range(len(results)):
        all_func_values = np.array(results[i][1:])
        min_func_values = np.min(all_func_values, axis=0)
        max_func_values = np.max(all_func_values, axis=0)
        mean_func_values = np.mean(all_func_values, axis=0)
        iterations = np.arange(len(results[i][0]))
        ax.fill_between(
            iterations, min_func_values, max_func_values, alpha=0.2
        )
        ax.plot(iterations, mean_func_values,  alpha=0.7)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value of the function")
    ax.set_title("Ribbon Plot of the function values from iterations")
    ax.legend()
    plt.savefig("ribbon_f9_max_iter.png")
    plt.show()  


class symulation_results:
    def __init__(self, length):
        self.optim_results = [[] for _ in range(length)]
        self.sum = [0] * length

    def symulation_mean(self, i):
        return self.sum[i] / len(self.optim_results[i])

    def add_new_result(self, optim_result, i):
        self.optim_results[i].append(optim_result)
        self.sum[i] += optim_result[-1]


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

    def update_mutation_stren(self, successful_mutations, i, mutation_stren) -> float:
        if i % self.adapt_interval == 0:
            if successful_mutations / self.adapt_interval > self.target_prob:
                mutation_stren *= 1.22
            elif successful_mutations / self.adapt_interval < self.target_prob:
                mutation_stren *= 0.82
            successful_mutations = 0
        return successful_mutations, mutation_stren


class Evolutionary:
    def __init__(self, function, x0, params, mut_stren_adapt_policy):
        self.f = function
        self.params = params
        self.success_mutations = 0
        self.result = optim_result(x0, function(x0), params.max_iter, [function(x0)])
        self.point_size = len(x0)
        self.mut_stren_adapt_policy = mut_stren_adapt_policy

    def mutate(self, mut_point, mutation_stren):
        return [
            self.result.min_point[i] + mutation_stren * mut_point[i]
            for i in range(self.point_size)
        ]

    def update_min_point(self, mut_point):
        if self.result.min_func_value >= self.f(mut_point):
            self.success_mutations += 1
            self.result.min_func_value = self.f(mut_point)
            self.result.min_point = mut_point
        self.result.func_values.append(self.result.min_func_value)

    def evolution_algorithm(self) -> optim_result:
        for i in range(self.params.max_iter):
            random_point = [random.gauss(0, 1) for _ in range(self.point_size)]
            mut_point = self.mutate(random_point, self.params.mutation_stren)
            self.update_min_point(mut_point)
            (
                self.success_mutations,
                self.params.mutation_stren,
            ) = self.mut_stren_adapt_policy.update_mutation_stren(
                self.success_mutations, i, self.params.mutation_stren
            )


if __name__ == "__main__":
    x0 = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    mut_stren_adapt_policy = one_fifth_succ_rule(10.0)
    # must_stren_list = [one_fifth_succ_rule(0.005), one_fifth_succ_rule(1.0), one_fifth_succ_rule(50.0), one_fifth_succ_rule(1000.0)]
    # params_list = [params_t(0.1, 10000), params_t(0.1, 10000), params_t(0.1, 10000), params_t(0.1, 10000)]
    params_list = [params_t(0.1, 10000)]
    results = symulation_results(len(params_list))
    for i in range(len(params_list)):
        for _ in range(10):
            eval = Evolutionary(f9, x0, params_list[0], mut_stren_adapt_policy)
            eval.evolution_algorithm()
            results.add_new_result(eval.result.func_values, i)
            print(_,"Minimum dla f1:",eval.result.min_point,"Wartość funkcji celu:",eval.result.min_func_value)
        print("Średnia: ", results.symulation_mean(i))

    make_ribbon_plot(results.optim_results)
    make_plot(results.optim_results)

