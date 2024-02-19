import random
import numpy as np
import math
import time

def distance(city1, city2):
    # Compute Euclidean distance between two cities
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def stsp_genetic_algorithm(cities, population_size=50, generations=1000, crossover_rate=0.8, mutation_rate=0.2):
    start_time = time.time() # Recording start time

    n = len(cities)
    best_tour = None
    best_distance = float('inf')
    
    
    # Create initial population
    population = [random.sample(range(1, n), n - 1) for _ in range(population_size)]  # Exclude city 0 initially
    
    # Main loop
    for gen in range(generations):
        
        
        # Add city 0 at the beginning and end of each tour
        population = [[0] + tour + [0] for tour in population]
        
        # Evaluate fitness of each individual
        fitness_values = [tour_fitness(tour, cities) for tour in population]
        
        # Find the best tour in the current population
        min_fitness_index = np.argmin(fitness_values)
        if fitness_values[min_fitness_index] < best_distance:
            best_tour = population[min_fitness_index]
            best_distance = fitness_values[min_fitness_index]
        
        # Selection: Roulette wheel selection
        selected_indices = roulette_wheel_selection(fitness_values, population_size)
        
        # Crossover
        offspring = []
        for i in range(0, population_size, 2):
            parent1, parent2 = population[selected_indices[i]], population[selected_indices[i+1]]
            if random.random() < crossover_rate:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]  # Create copies
            offspring.extend([child1, child2])
        
        # Mutation: Swap mutation
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                mutate(offspring[i])
        
        # Update population with offspring
        population = offspring
    
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time

    return best_tour, best_distance, execution_time

def tour_fitness(tour, cities):
    return sum(distance(cities[tour[i]], cities[tour[i+1]]) for i in range(len(tour)-1)) + distance(cities[tour[-1]], cities[tour[0]])

def roulette_wheel_selection(fitness_values, population_size):
    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]
    selected_indices = np.random.choice(range(len(fitness_values)), size=population_size, replace=True, p=probabilities)
    return selected_indices

def order_crossover(parent1, parent2):
    n = len(parent1)
    start = random.randint(0, n-1)
    end = random.randint(start+1, n)
    child1, child2 = [-1] * n, [-1] * n
    
    # Copy the segment between start and end from parent1 to child1, and from parent2 to child2
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]
    
    # Fill in the remaining positions in child2 with the remaining unused cities from parent1
    # Fill in the remaining positions in child1 with the remaining unused cities from parent2
    for i in range(n):
        if parent2[i] not in child1:
            idx = child1.index(-1)
            child1[idx] = parent2[i]
        if parent1[i] not in child2:
            idx = child2.index(-1)
            child2[idx] = parent1[i]
    
    return child1, child2

def mutate(tour):
    n = len(tour)
    idx1, idx2 = random.sample(range(n), 2)
    tour[idx1], tour[idx2] = tour[idx2], tour[idx1]

def read_tsp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    cities = []
    edge_weight_section = False
    for line in lines:
        line = line.strip()
        if line.startswith('EOF'):
            break
        if edge_weight_section:
            city = tuple(map(int, line.split()))
            cities.append(city)
        if line.startswith('EDGE_WEIGHT_SECTION'):
            edge_weight_section = True

    return cities

# Read city coordinates from the TSP file
tsp_file_path = 'Project 2\gr17.tsp'
cities = read_tsp_file(tsp_file_path)

# Solve TSP using Branch and Bound
best_tour, best_distance, execution_time = stsp_genetic_algorithm(cities)

print('Solving STSP using Genetic Algorithm:')
print("Optimal Tour:", best_tour)
print("Optimal Distance:", best_distance)
print("Execution Time:", execution_time, "seconds")

def distance(city1, city2, cities):
    # Return the distance between city1 and city2
    return cities[city1][city2]

def tour_fitness(tour, cities):
    # Calculate the total distance of the tour
    total_distance = sum(distance(tour[i], tour[i+1], cities) for i in range(len(tour) - 1))
    # Add the distance from the last city back to the starting city
    total_distance += distance(tour[-1], tour[0], cities)
    return total_distance

def read_astp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    dimension = None
    edge_weight_section = False
    distance_matrix = []

    for line in lines:
        line = line.strip()
        if line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1])
            distance_matrix = [[0] * dimension for _ in range(dimension)]
        elif line.startswith('EDGE_WEIGHT_SECTION'):
            edge_weight_section = True
        elif edge_weight_section and line != 'EOF':
            distances = list(map(int, line.split()))
            for i in range(len(distances)):
                distance_matrix[len(distance_matrix) - len(distances) + i][:len(distances)] = distances[:]
    return distance_matrix


def pmx_crossover(parent1, parent2):
    n = len(parent1)
    start = random.randint(0, n - 2)
    end = random.randint(start + 1, n - 1)
    child1, child2 = [-1] * n, [-1] * n
    
    # Copy the segment between start and end from parent1 to child1
    child1[start:end] = parent1[start:end]
    
    # Copy the segment between start and end from parent2 to child2
    child2[start:end] = parent2[start:end]
    
    # Map elements from parent2 to child1 and from parent1 to child2
    for i in range(start, end):
        if parent2[i] not in child1:
            idx = parent2.tolist().index(parent1[i])
            while start <= idx < end:
                idx = parent2.tolist().index(parent1[idx])
            child1[idx] = parent2[i]
        if parent1[i] not in child2:
            idx = parent1.tolist().index(parent2[i])
            while start <= idx < end:
                idx = parent1.tolist().index(parent2[idx])
            child2[idx] = parent1[i]
    
    # Fill in the remaining positions in child1 with the remaining unused elements from parent2
    for i in range(n):
        if child1[i] == -1:
            child1[i] = parent2[i]
    
    # Fill in the remaining positions in child2 with the remaining unused elements from parent1
    for i in range(n):
        if child2[i] == -1:
            child2[i] = parent1[i]
    
    return child1, child2

def atsp_genetic_algorithm(cities, population_size=50, generations=1000, crossover_rate=0.8, mutation_rate=0.2):
    start_time = time.time()  # Record the start time

    n = len(cities)
    best_tour = None
    best_distance = float('inf')
    
    # Main loop
    for gen in range(generations):
        
        # Create initial population
        population = [np.random.permutation(n) for _ in range(population_size)]
        
        # Evaluate fitness of each individual
        fitness_values = [tour_fitness(tour, cities) for tour in population]
        
        # Find the best tour in the current population
        min_fitness_index = np.argmin(fitness_values)
        if fitness_values[min_fitness_index] < best_distance:
            best_tour = population[min_fitness_index]
            best_distance = fitness_values[min_fitness_index]
        
        # Selection: Roulette wheel selection
        selected_indices = roulette_wheel_selection(fitness_values, population_size)
        
        # Crossover
        offspring = []
        for i in range(0, population_size, 2):
            parent1, parent2 = population[selected_indices[i]], population[selected_indices[i+1]]
            if np.random.rand() < crossover_rate:
                child1, child2 = pmx_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()  # Create copies
            offspring.extend([child1, child2])
        
        # Mutation: Swap mutation
        for i in range(len(offspring)):
            if np.random.rand() < mutation_rate:
                mutate(offspring[i])
        
        # Update population with offspring
        population = offspring
    
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time

    return best_tour, best_distance, execution_time


# Read city coordinates from the ATSP file
atsp_file_path = 'Project 2/br17.atsp'
cities = read_astp_file(atsp_file_path)

# Solve ATSP using Genetic Algorithm
best_tour, best_distance, execution_time = atsp_genetic_algorithm(cities)
print('Solving ATSP using Genetic Algorithm:')

print("Optimal Tour:", best_tour)
print("Optimal Distance:", best_distance)
print("Execution Time:", execution_time, "seconds")

