import math

def distance(city1, city2):
    # Compute Euclidean distance between two cities
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def nearest_neighbor(cities):
    n = len(cities)
    unvisited = set(range(n))
    current_city = 0  # Start at city 0
    tour = [current_city]  # Initialize tour with starting city
    
    unvisited.remove(current_city)
    
    print("Step 0: Start at city 0")
    
    step = 1
    
    while unvisited:
        nearest_city = min(unvisited, key=lambda city: distance(cities[current_city], cities[city]))
        tour.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city
        
        print(f"Step {step}: Visit city {nearest_city}")
        print(f"Current Tour: {tour}")
        step += 1
    
    # Complete the tour by returning to the starting city
    tour.append(tour[0])
    
    # Compute the total distance of the tour
    total_distance = sum(distance(cities[tour[i]], cities[tour[i+1]]) for i in range(n))
    
    return tour, total_distance

# Example usage:
cities = [(0, 0), (1, 2), (3, 1), (2, 3)]
best_tour, best_distance = nearest_neighbor(cities)
print("Optimal Tour:", best_tour)
print("Optimal Distance:", best_distance)
