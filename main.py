import numpy as np
from cec2017.functions import f1, f9
import matplotlib.pyplot as plt
import random

def make_plot(results):
    for idx, result in enumerate(results[1:]):
        plt.plot(
            [i for i in range(0, result.num_of_iterations + 1)],
            result.func_values,
            label=f"Result {idx+1}",
        )
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Value of the objective function")
    plt.title("Plot of the objective function values from iterations")
    plt.legend()
    plt.show()
    
def make_ribbon_plot(results):
    _, ax = plt.subplots()
    all_func_values = np.array(results[1:])
    min_func_values = np.min(all_func_values, axis=0)
    max_func_values = np.max(all_func_values, axis=0)
    mean_func_values = np.mean(all_func_values, axis=0)
    iterations = np.arange(len(results[0]))
    ax.fill_between(iterations, min_func_values, max_func_values, alpha=0.2)
    ax.plot(iterations, mean_func_values, color='b', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value of the objective function")
    ax.set_title("Ribbon Plot of the objective function values from iterations")
    ax.legend()
    plt.show()


class symulation_results:
    def __init__(self):
        self.optim_results = []
        self.sum = 0
        
    def mean(self):
        return self.sum/len(self.optim_results)
    
    def add_new_result(self, optim_result):
        self.optim_results.append(optim_result)
        self.sum += optim_result[-1]
    

class optim_result:
    def __init__(self, minimum, min_func_value, num_of_iterations=1, func_values=[]):
        self.minimum = minimum
        self.min_func_value = min_func_value
        self.num_of_iterations = num_of_iterations
        self.func_values = func_values

class params_t:
    def __init__(self, sigma, max_iter):
        self.sigma = sigma
        self.max_iter = max_iter

class one_fifth_succ_rule:
    def __init__(self, adaptation_interval):
        self.adaptation_interval = adaptation_interval
        self.target_prob = 1/5

    def update_mutation_stren(self, successful_mutations, i, mutation_stren) -> float:
        if i % self.adaptation_interval == 0:
            if successful_mutations / self.adaptation_interval > self.target_prob:
                mutation_stren *= 1.22
            elif successful_mutations / self.adaptation_interval < self.target_prob:
                mutation_stren *= 0.82
            successful_mutations = 0
        return successful_mutations, mutation_stren

class Evolutionary:
    def __init__(self, function, x0, params, stepsize_adaptation_policy):
        self.f = function
        self.params = params
        self.successful_mutations = 0
        self.result = optim_result(x0, function(x0), params.max_iter, [function(x0)])
        self.population_size = len(x0)
        self.stepsize_adaptation_policy = stepsize_adaptation_policy


    def mutate(self, reproduction, mutation_rate):
        return [self.result.minimum[i] + mutation_rate * reproduction[i] for i in range(self.population_size)]
        # return np.sum(population + np.multiply(reproduction, mutation_rate))
        # numpy sum i multiply

    def update_min_point(self, mut_population):
        if self.result.min_func_value >= self.f(mut_population):
            self.successful_mutations += 1
            self.result.min_func_value = self.f(mut_population)
            self.result.minimum = mut_population
        self.result.func_values.append(self.result.min_func_value)
         
            
    def evolution_algorithm(self) -> optim_result:
        for i in range(self.params.max_iter):
            random_point = [random.gauss(0, 1) for _ in range(self.population_size)]
            mut_population = self.mutate(random_point, self.params.sigma)
            self.update_min_point(mut_population)
            self.successful_mutations, self.params.sigma = self.stepsize_adaptation_policy.update_mutation_stren(self.successful_mutations, i, self.params.sigma)


if __name__ == "__main__":
    x0 = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    stepsize_adapt_policy = one_fifth_succ_rule(30.0)
    params = params_t(10.0, 100000)
    wyniczki = symulation_results()
    for _ in range(3):
        eval = Evolutionary(f1, x0, params, stepsize_adapt_policy)
        eval.evolution_algorithm()
        wyniczki.add_new_result(eval.result.func_values)
        print(_, "Minimum dla f1:", eval.result.minimum, "Wartość funkcji celu:", eval.result.min_func_value)
    print("Średnia: ", wyniczki.mean())
    make_ribbon_plot(wyniczki.optim_results)