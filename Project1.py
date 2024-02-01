import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time

# Benchmark Functions
def schwefel(x):
    return 418.9829*2 - x[0] * np.sin(np.sqrt(np.abs(x[0]))) - x[1] * np.sin(np.sqrt(np.abs(x[1])))

def de_jong_1(x):
    return sum(x ** 2)

def rosenbrocks_saddle(x):
    return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x):
    return 10 * len(x) + sum(x**2 - 10 * np.cos(2 * np.pi * x))

def griewangk(x):
    return sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1
    
#added the other 5 functions
def sine_envelope_sine(x):
    return - sum(0.5 + ((np.sin(x[i]**2 + x[i+1]**2 - 0.5)**2)/(1 + 0.001 * (x[i]**2 + x[i+1]**2))**2))

def stretched_V_sine(x):
    return sum(([i]**2 + x[i+1]**2)**0.25 * np.sin(50 * (x[i]**2 + x[i+1]**2)**0.1)**2 + 1)

def ackleys_one(x):
    return sum((1/(math.e**0.2)) * (np.sqrt(x[i]**2 + x[i+1]**2)) + 3 * (np.cos(2 * x[i]) + np.sin(2 * x[i+1])))

def ackleys_two(x):
    return sum(20 + math.e - (20/(math.e**(0.2 * np.sqrt((x[i]**2 + x[i+1]**2)/2)))) - math.e**(0.5 * (np.cos((2 * np.pi) * x[i])) + np.cos((2 * np.pi) * x[i+1])))

def egg_holder(x):
    return sum(-x[i] * np.sin(np.sqrt(np.abs(x[i] - x[i+1] - 47))) - (x[i+1] + 47) * np.sin(np.sqrt(np.abs(x[i+1] + 47 + (x[i]/2)))))

# Generate Pseudo-random Solution Vectors
def generate_solution_vectors(method, dim=30, num_vectors=30):
    if method == "mersenne_twister":
        np.random.seed(42)  # DELETE once done testing!!!!!!
 
        # Generate solution vectors between -500 and 500
        solution_vectors = np.random.randint(-500, 501, (num_vectors, dim), dtype=np.int32)
        
        print("Random Solution Vectors:")
        print(solution_vectors)

        # Cast the solution vectors to float64 for calculations
        return solution_vectors.astype(np.float64)
    else:
        raise ValueError("Invalid method.")

# Solve Benchmark Functions for Solution Vectors
def solve_functions(solution_vectors):
    
    functions = [schwefel, de_jong_1, rosenbrocks_saddle, rastrigin, griewangk]
    
    # Initialize an array to store the results of function evaluations
    results = np.zeros((len(functions), len(solution_vectors)), dtype=np.float64)
    
    # Iterate over each function and each solution vector
    for i, func in enumerate(functions):
        
        for j, solution in enumerate(solution_vectors):
            # Evaluate the function for the current solution vector
            results[i, j] = func(solution)

    return results

# Function to compute Statistical Analysis
def compute_statistics(results):
    
    statistics = {
        "average": np.mean(results, axis=1),
        "standard deviation": np.std(results, axis=1),
        "range": np.ptp(results, axis=1),
        "median": np.median(results, axis=1)
    }

    return statistics

# Generate solution vectors using the Mersenne Twister method
# dim: The dimensionality of each solution vector 
# num_vectors: The number of solution vectors to generater
solution_vectors = generate_solution_vectors(method="mersenne_twister", dim=30, num_vectors=30)

# Capture start time before solving the benchmark functions
start_time = time.time()

# Solve the benchmark functions for the generated solution vectors
results = solve_functions(solution_vectors)

# Capture end time after solving the benchmark functions
end_time = time.time()

# Compute statistical analysis 
statistics = compute_statistics(results)

# Print stats
print("\nStatistics:")
for stat, values in statistics.items():
   
    print(f"{stat}: {values}")

print(f"Time taken: {end_time - start_time} seconds")
