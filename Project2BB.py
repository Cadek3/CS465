import heapq

# Function definition to solve STSP using branch and bound
# Explore all possible cities to add to the current tour
def stsp_branch_and_bound(cities):
    # Define distance function within stsp_branch_and_bound
    def distance(city1, city2):
        # Compute distance between two cities (Euclidean distance)
        return ((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2) ** 0.5
    
    n = len(cities)
    best_tour = None
    best_tour_distance = float('inf')
    
    # For complete tours, calculate the total distance of the tour
    # by summing distances between consecutive cities
    # For partial tours, consider all possible edges 
    # between the current city and the remaining cities
    def lower_bound(tour):
        if len(tour) == n:  # Complete tour
            return sum(distance(cities[tour[i]], cities[tour[i+1]]) for i in range(n-1)) + distance(cities[tour[-1]], cities[tour[0]])
        else:
            return sum(distance(cities[tour[i]], cities[tour[i+1]]) for i in range(len(tour)-1))
    
    # Initialize priority queue (min-heap) with initial tours of length 2
    pq = [(lower_bound([i, j]), [i, j]) for i in range(n) for j in range(n) if i != j]
    
    # Counter for step number
    step = 1
    
    while pq:
        # Pop the tour with the smallest lower bound
        lb, tour = heapq.heappop(pq)
        
        # Print information about the current step
        print(f"Step {step}: Tour = {tour}, Lower Bound = {lb}")
        step += 1
        
        if lb >= best_tour_distance:
            continue  # Prune this branch if the lower bound exceeds the current best distance
        if len(tour) == n:  # Complete tour found
            tour_distance = sum(distance(cities[tour[i]], cities[tour[i+1]]) for i in range(n-1)) + distance(cities[tour[-1]], cities[tour[0]])
            if tour_distance < best_tour_distance:
                best_tour_distance = tour_distance
                best_tour = tour
        else:
            # Expand the tour by adding one more city
            for city in set(range(n)) - set(tour):
                new_tour = tour + [city]
                heapq.heappush(pq, (lower_bound(new_tour), new_tour))
    
    return best_tour, best_tour_distance

# Function definition to solve ATSP using branch and bound
# Explore all possible cities to add to the current tour, ensuring each city is added only once
def atsp_branch_and_bound(cities):
    # Define distance function within atsp_branch_and_bound
    def distance(city1, city2):
        # Compute distance between two cities (Euclidean distance)
        return ((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2) ** 0.5
    
    n = len(cities)
    best_tour = None
    best_tour_distance = float('inf')
    
    # For complete tours, calculate the total distance of the 
    # tour by summing distances between consecutive cities
    # For partial tours, consider only the minimum distance 
    # from each city to any other city in the remaining set
    def lower_bound(tour):
        if len(tour) == n:  # Complete tour
            return sum(distance(cities[tour[i]], cities[tour[i+1]]) for i in range(n-1)) + distance(cities[tour[-1]], cities[tour[0]])
        else:
            return sum(min(distance(cities[tour[i]], cities[j]) for j in range(n) if j != tour[i]) for i in range(len(tour)-1))
    
    # Initialize priority queue (min-heap) with initial tours of length 2
    pq = [(lower_bound([i, j]), [i, j]) for i in range(n) for j in range(n) if i != j]
    
    # Counter for step number
    step = 1
    
    while pq:
        # Pop the tour with the smallest lower bound
        lb, tour = heapq.heappop(pq)
        
        # Print information about the current step
        print(f"Step {step}: Tour = {tour}, Lower Bound = {lb}")
        step += 1
        
        if lb >= best_tour_distance:
            continue  # Prune this branch if the lower bound exceeds the current best distance
        if len(tour) == n:  # Complete tour found
            tour_distance = sum(distance(cities[tour[i]], cities[tour[i+1]]) for i in range(n-1)) + distance(cities[tour[-1]], cities[tour[0]])
            if tour_distance < best_tour_distance:
                best_tour_distance = tour_distance
                best_tour = tour
        else:
            # Expand the tour by adding one more city
            for city in set(range(n)) - set(tour):
                new_tour = tour + [city]
                heapq.heappush(pq, (lower_bound(new_tour), new_tour))
    
    return best_tour, best_tour_distance



cities = [(0, 0), (1, 2), (3, 1), (2, 3)]

def print_cities(cities):
    print("City Coordinates:")
    for i, city in enumerate(cities):
        print(f"City {i}: {city}")

# Print city coordinates
print_cities(cities)
print()
# Using Branch and Bound to Solve STSP
print("Applying Branch and Bound to solve STSP. . .\n")
best_tour, best_distance = stsp_branch_and_bound(cities)
print()
print("Optimal Tour:", best_tour)
print("Optimal Distance:", best_tour + [best_tour[0]])
print()

# Using Branch and Bound to Solve ATSP
print("Applying Branch and Bound to solve ATSP. . .\n")
cities = [(0, 0), (1, 2), (3, 1), (2, 3)]

best_tour, best_distance = atsp_branch_and_bound(cities)

print()
print("Optimal Tour:", best_tour + [best_tour[0]])
print("Optimal Distance:", best_distance)
