import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time

# Benchmark Functions
def schwefel(x):
    n = len(x)
    sum_term = np.sum(-x * np.sin(np.abs(x)**0.5))
    result = 418.9829 * n - sum_term
    return result


def de_jong_1(x):
    return np.sum(x**2)

def rosenbrocks_saddle(x):
    n = len(x)
    result = 0
    for i in range(n - 1):
        term = 100 * (x[i]**2 - x[i+1])**2 + (1 - x[i])**2
        result += term
    return result


def rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def griewangk(x):
    n = len(x)
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
    return 1 + sum_term - prod_term
    
#added the other 5 functions
def sine_envelope_sine(x):
    n = len(x)
    result = 0
    for i in range(n - 1):
        term = 0.5 + np.sin(x[i]**2 + x[i+1]**2 - 0.5)**2
        term /= 1 + 0.001 * (x[i]**2 + x[i+1]**2)**2
        result -= term
    return result

def stretch_v_sine_wave(x):
    result = 0
    n = len(x)
    for i in range(n - 1):
        term = (np.sqrt((x[i]**2 + x[i+1]**2))**(1/4)) * (np.sin(50 * (np.sqrt((x[i]**2 + x[i+1]**2)))**(1/10)))**2
        result += term
    return result + 1

def ackley_one(x):
    result = 0
    n = len(x)
    for i in range(n - 1):
        term = np.exp(0.2) * (np.sqrt(x[i]**2 + x[i+1]**2)) + 3 * (np.cos(2 * x[i]) + np.sin(2 * x[i+1]))
        result += term
    return result

def ackley_two(x):
    n = len(x)
    sum_term = 0
    for i in range(n - 1):
        sum_term += 20 + np.exp(1) - (20 / np.exp(0.2 * np.sqrt(x[i]**2 + x[i+1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[i]) + np.cos(2 * np.pi * x[i+1])))
    return sum_term

def egg_holder(x):
    result = 0
    n = len(x)
    for i in range(n - 1):
        result += -x[i] * np.sin(np.sqrt(np.abs(x[i] - x[i+1] - 47))) - (x[i+1] + 47) * np.sin(np.sqrt(np.abs(x[i+1] + 47 + x[i]/2)))
    return result

# Generate random number vector using LCG method
def linear_congruential_generator(seed, multi, inc, modu, dim):
    random = []
    current = seed

    for _ in range(dim):
        current = (multi * current + inc) % modu
        new = 10 + (current / modu) * (100 - 10)
        random.append(new)

    return np.array(random)


# Solve Benchmark Functions for Solution Vectors
def solve_functions(solution_vectors):
    functions = [schwefel, de_jong_1, rosenbrocks_saddle, rastrigin, griewangk,
                 sine_envelope_sine, stretch_v_sine_wave, ackley_one, ackley_two, egg_holder]

    # Initialize an array to store the results of function evaluations
    results = np.zeros((len(functions), len(solution_vectors)), dtype=np.float64)
    times = np.zeros((len(functions),), dtype=np.float64)
    
    # Iterate over each function and each solution vector
    for i, func in enumerate(functions):
        start_time = time.time()  # Capture start time
        for j, solution in enumerate(solution_vectors):
            # Evaluate the function for the current solution vector
            results[i, j] = func(solution)
        end_time = time.time()  # Capture end time
        times[i] = (end_time - start_time) * 1000  # Convert to milliseconds and store time taken
    
    return results, times



# Function to compute Statistical Analysis
def compute_statistics(results):
   
    functions = ["Schwefel", "De Jong 1", "Rosenbrock's Saddle", "Rastrigin", "Griewangk",
                 "Sine Envelope Sine Wave", "Stretch V Sine Wave", "Ackley One", "Ackley Two", "Egg Holder"]
    
    statistics = {}
    
    # Compute statistics for each function
    for i, func_name in enumerate(functions):
        func_results = results[i]
        stats = {
            "average": np.mean(func_results),
            "standard deviation": np.std(func_results),
            "range": np.ptp(func_results),
            "median": np.median(func_results)
        }
        statistics[func_name] = stats

    return statistics

# Generate solution vectors using the Linear Congruential Generator (LCG) method
current_time_seed = int(time.time())
random_vectors = [linear_congruential_generator(seed=current_time_seed + i, multi=1664525, inc=1013904223, modu=2**32, dim=30) for i in range(30)]


# Solve the benchmark functions for the generated solution vectors
results, times = solve_functions(random_vectors)


# Compute statistical analysis 
statistics = compute_statistics(results)

# Print stats
print("\nStatistics:")
for func_name, stats in statistics.items():
    print(f"{func_name}:")
    for stat, value in stats.items():
        print(f"{stat}: {value}")
    print()

# Print time taken for each function
print("\nTime taken for each function (milliseconds):")
functions = ["Schwefel", "De Jong 1", "Rosenbrock's Saddle", "Rastrigin", "Griewangk",
             "Sine Envelope Sine Wave", "Stretch V Sine Wave", "Ackley One", "Ackley Two", "Egg Holder"]
for func_name, time_value in zip(functions, times):
    print(f"{func_name}: {time_value} milliseconds")
