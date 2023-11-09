import autograd.numpy as np
from functools import partial
from cec2017 import functions
import matplotlib.pyplot as plt
from autograd import grad


class optim_result_t:
    def __init__(self, minimum, min_func_value, num_of_iterations):
        self.minimum = minimum
        self.min_func_value = min_func_value
        self.num_of_iterations = num_of_iterations


class params_t:
    def __init__(
        self,
        learning_rate,
        epsilon,
        max_iter,
        max_iter_flag,
        epsilon_flag,
        convergence_flag,
    ):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.max_iter_flag = max_iter_flag
        self.epsilon_flag = epsilon_flag
        self.convergence_flag = convergence_flag


def make_plot(result):
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


def objective_function(x, alpha):
    n = len(x)
    i = np.arange(n) + 1
    if (n - 1) == 0:
        x0 = x
    else:
        x0 = (alpha ** ((i - 1) / (n - 1))) * (x**2)
    return x0.sum()


objective_function_1 = partial(objective_function, alpha=1.0)
objective_function_10 = partial(objective_function, alpha=10.0)
objective_function_100 = partial(objective_function, alpha=100.0)


def gradient_descent(obj_function, x0, params) -> optim_result_t:
    x = x0.copy()
    obj_func_values = []
    i = 0
    while True:
        obj_func_values.append(obj_function(x=x))
        gradient_func = grad(obj_function)
        gradient_value = gradient_func(x)
        if (
            np.linalg.norm(gradient_value) <= params.epsilon
            and params.epsilon_flag == 1
        ):
            break
        x = x - params.learning_rate * gradient_value
        if (
            params.convergence_flag == 1
            and np.max(np.abs(params.learning_rate * gradient_value)) < params.epsilon
        ):
            break
        if params.max_iter_flag == 1 and i >= params.max_iter:
            break
        i += 1
    return optim_result_t(x, obj_func_values, i)


if __name__ == "__main__":
    x0 = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    beta = 1e-8
    params = [
        params_t(beta, 1e-6, 100, 1, 1, 1),
        params_t(beta, 1e-6, 100, 1, 1, 1),
        params_t(beta, 1e-6, 100, 1, 1, 1),
        params_t(beta, 1e-6, 100, 1, 1, 1),
    ]
    results = [gradient_descent(functions.f1, x0, param)
               for param in params]

    make_plot(results)
    print(results[-1].min_func_value[-1])