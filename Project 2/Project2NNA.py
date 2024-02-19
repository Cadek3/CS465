
import math
import time

def distance(city1, city2):
    # Compute Euclidean distance between two cities
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def stsp_nearest_neighbor(cities):
    start_time = time.time()  # Record the start time

    n = len(cities)
    unvisited = set(range(n))
    
    # Find the city with the minimum x-coordinate as the starting city
    start_city = min(unvisited, key=lambda city: cities[city][0])
    current_city = start_city
    tour = [current_city]  # Initialize tour with the starting city
    
    unvisited.remove(current_city)
    
    step = 1
    
    while unvisited:
        nearest_city = min(unvisited, key=lambda city: distance(cities[current_city], cities[city]))
        tour.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city
        step += 1
    
    # Complete the tour by returning to the starting city
    tour.append(start_city)
    
    # Compute the total distance of the tour
    total_distance = sum(distance(cities[tour[i]], cities[tour[i+1]]) for i in range(n))
    
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time

    return tour, total_distance, step - 1, execution_time  # Return the total steps taken

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

# Example usage for STSP
tsp_file_path = 'Project 2\gr17.tsp'  # Adjust the file path accordingly
cities = read_tsp_file(tsp_file_path)
print('Solving STSP using Nearest Neighbor')
best_tour, best_distance, total_steps, execution_time = stsp_nearest_neighbor(cities)
print("Optimal Tour:", best_tour)
print("Optimal Distance:", best_distance)
print("Total Steps:", total_steps)
print("Execution Time:", execution_time, "seconds")

def distance(city1, city2, cities):
    # Return the distance between city1 and city2
    return cities[city1][city2]

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

def atsp_nearest_neighbor(cities):
    start_time = time.time()  # Record the start time    
    
    n = len(cities)
    unvisited = set(range(n))
    
    start_city = 0  # Start from the first city
    current_city = start_city
    tour = [current_city]  # Initialize tour with the starting city
    unvisited.remove(current_city)
    
    step = 1
    while unvisited:
        nearest_city = min(unvisited, key=lambda city: cities[current_city][city])
        tour.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city
        step += 1
    
    # Complete the tour by returning to the starting city
    tour.append(start_city)
    
    # Compute the total distance of the tour
    total_distance = sum(cities[tour[i]][tour[i+1]] for i in range(n))
    
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time

    return tour, total_distance, step - 1, execution_time

print()
print('Solving ATSP using Nearest Neighbor')

# Example usage:
astp_file_path = 'Project 2/br17.atsp'  # Adjust the file path accordingly
distance_matrix = read_astp_file(astp_file_path)
optimal_tour, optimal_distance, total_steps, execution_time = atsp_nearest_neighbor(distance_matrix)
print("Optimal Tour:", optimal_tour)
print("Optimal Distance:", optimal_distance)
print("Total Steps:", total_steps)
print("Execution Time:", execution_time, "seconds")


