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

# Generate Pseudo-random Solution Vectors
def generate_solution_vectors(method, dim=30, num_vectors=30):
    if method == "mersenne_twister":
        np.random.seed(42)  # chosen seed
 
        # Generate solution vectors between -500 and 500
        solution_vectors = np.random.randint(-500, 501, (num_vectors, dim), dtype=np.int32)

        
        # Cast the solution vectors to float64 for calculations
        return solution_vectors.astype(np.float64)
    else:
        raise ValueError("Invalid method.")

# Solve Benchmark Functions for Solution Vectors
def solve_functions(solution_vectors):
    
    functions = [schwefel, de_jong_1, rosenbrocks_saddle, rastrigin, griewangk,
                 sine_envelope_sine, stretch_v_sine_wave, ackley_one, ackley_two, egg_holder]
    
    # Initialize an array to store the results of function evaluations
    results = np.zeros((len(functions), len(solution_vectors)), dtype=np.float64)
    times = np.zeros((len(functions),), dtype=np.float64)
    
    # Iterate over each function and each solution vector
    for i, func in enumerate(functions):
        start_time = time.time()
        for j, solution in enumerate(solution_vectors):
            # Evaluate the function for the current solution vector
            results[i, j] = func(solution)
        end_time = time.time()
        times[i] = (end_time - start_time) * 1000  # Convert to milliseconds

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

# Generate solution vectors using the Mersenne Twister method
solution_vectors = generate_solution_vectors(method="mersenne_twister", dim=30, num_vectors=30)

# Solve the benchmark functions for the generated solution vectors
results, times = solve_functions(solution_vectors)

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

print()
# Benchmark Functions
def schwefel_optimized(x):
    return np.sum(-x * np.sin(np.abs(x)**0.5))

print("Attempting Gradient Descent on Schwefel")
# ATTEMPTING
# Gradient Descent Function
def gradient_descent(start_x, learning_rate, num_iterations, df_dx, f):
    x = start_x
    history = []
    for i in range(num_iterations):
        grad_x = df_dx(x)
        x = x - learning_rate * grad_x
        history.append((x, f(x)))
    return x, f(x), history

# Define the objective function (Schwefel function)
def f(x):
    return schwefel_optimized(x)

# Define the derivative of the Schwefel function
def df_dx_schwefel(x):
    return 2 * x * np.sin(np.abs(x)**0.5) - np.sign(x) * np.cos(np.abs(x)**0.5) * np.sqrt(np.abs(x))

# Apply gradient descent to optimize the Schwefel function
start_x = 400
learning_rate = 0.01
num_iterations = 1000

final_x, min_value, history = gradient_descent(start_x, learning_rate, num_iterations, df_dx_schwefel, f)

print("Optimized value of x:", final_x)
print("Minimum value:", min_value)
