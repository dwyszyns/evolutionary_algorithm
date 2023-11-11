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
    fig, ax = plt.subplots()

    for idx, result in enumerate(results[1:]):
        iterations = np.arange(result.num_of_iterations + 1)
        func_values = result.func_values
        ax.fill_between(iterations, min(func_values), max(func_values), alpha=0.2, label=f"Result {idx+1}")

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value of the objective function")
    ax.set_title("Ribbon Plot of the objective function values from iterations")
    ax.legend()
    plt.show()


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
    

# można dodać parametr, który sprawdza średnią
# epsilon
# ile licz sprawdzamy

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
        zmienna = self.f(mut_population)
        if self.result.min_func_value >= zmienna:
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
        # return optim_result(self.result.minimum, self.result.min_func_value, params.max_iter)


if __name__ == "__main__":
    x0 = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    stepsize_adapt_policy = one_fifth_succ_rule(30.0)
    params = params_t(10.0, 100000)
    obiekty =[]
    #range max 10
    suma = 0
    for _ in range(10):
        eval = Evolutionary(
            f1, x0, params, stepsize_adapt_policy
        )
        eval.evolution_algorithm()
        # make_plot(eval.result)
        obiekty.append(eval.result)
        suma += eval.result.min_func_value
        print(_, "Minimum dla f1:", eval.result.minimum, "Wartość funkcji celu:", eval.result.min_func_value)
    print("Średnia: ", suma/10)
    make_plot(obiekty)
    make_ribbon_plot(obiekty)
        # make_plot(results)
    # print("wartosc funkcji celu:", result)
#rysować wykres funkcji czy średnią?

#wywalic population size