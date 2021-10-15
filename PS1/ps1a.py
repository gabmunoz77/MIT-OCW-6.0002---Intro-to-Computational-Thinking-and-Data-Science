###########################
# 6.0002 Problem Set 1a: Space Cows 
# Name: Gabriel Munoz
# Collaborators: None
# Time: Wednesday, April 28, 2021 - Tuesday, May 18, 2021

from ps1_partition import get_partitions
import time

#================================
# Part A: Transporting Space Cows
#================================

# Problem 1
def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """
    # Open the file for reading and extracting data
    cow_file = open(filename, 'r')
    # initialize empty dictionary to save data into
    cow_dict = {}
    # read file line by line
    for line in cow_file:
        # remove whitespace (spaces and newlines)
        line = line.strip()
        # make lists of each line to save data into <name, weight> pairs
        line = line.split(',')
        # want keys to be cow names (strings) and values to be cow ages (ints)
        # ASSUMPTIONS: cow names unique; weights will be numerical (?)
        cow_dict[line[0]] = int(line[1])
    # close file and return dictionary
    cow_file.close()
    return cow_dict

# Problem 2
def greedy_cow_transport(cows,limit=10):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # Create a list of key:value tuples sorted by value in descending order
    cows_sorted = sorted(cows.items(), key=lambda x: x[1], reverse=True)

    # Initialize empty list for cow trips
    cow_trips = []

    # Algo
    # 1) Iterate while loop as long as there are cows in cows_sorted
    # 2) In while loop, initialize empty list for current trip and current weight of trip
    # 3) Iterate through cows_sorted--for each tuple, if cow fits in ship, add to curr trip,
    #       update trip weight, and keep track of cows to remove from sorted list
    # 4) Once finished looping through cow_sorted once, append current trip to cow_trips, remove cows from sorted list,
    #       and while loop restarts with sorted cows list shorter
    # 5) Will exit while loop once no more cows are left to transport (cows_sorted is empty)

    while len(cows_sorted) != 0:
        curr_trip = []
        curr_weight = 0
        # to keep track of what cows to remove from sorted list
        cows_gone = []
        for cow in cows_sorted:
            if cow[1] + curr_weight <= limit:
                curr_trip.append(cow[0])
                curr_weight += cow[1]
                # to remove cows from sorted list after loop
                cows_gone.append(cow)
        cow_trips.append(curr_trip)
        # once a cow is added to a spaceship trip, it can't go again! It's already in Aurock!
        for cow in cows_gone:
            cows_sorted.remove(cow)
    return cow_trips

# Problem 3
def brute_force_cow_transport(cows,limit=10):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips 
        Use the given get_partitions function in ps1_partition.py to help you!
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # Create a list of key:value tuples, leaving dictionary intact
    cows_list = [(k, v) for k, v in cows.items()]

    # Enumerate all possible ways to divide cows in separate trips
    partitions = get_partitions(cows_list)

    # "Eliminate" partitions that contain a trip that exceeds the weight limit
    # keep track of the current minimum number of trips possible to transport all cows back to Aurock
    min_trips = limit
    # keep track of a potential optimal solution and update it as we find a new minimum number of trips
    an_optimal_solution = []
    # for each partition of the set of cows
    for partition in partitions:
        # bool to check if partition is a valid one
        valid = True
        # for each trip in a given partition
        for trip in partition:
            # keep track of the weight
            curr_weight = 0
            for cow in trip:
                curr_weight += cow[1]
            # if any trip weight exceeds limit in any given partition, that is an invalid partition
            if curr_weight > limit:
                valid = False
                # takes us to next partition (breaks out of current for loop)
                break
        # if valid partition, check for a new minimum number of trips, and if new min, update current optimal solution
        if valid and len(partition) < min_trips:
            min_trips = len(partition)
            # by the end of outermost for loop, this WILL be the optimal solution
            an_optimal_solution = partition

    # DON'T EVEN NEED TO STORE ALL THE VALID PARTITIONS--we just need to UPDATE our MINIMUM at every valid partition
    # and return the last partition we found with the min number of trips.
    # THIS IS LIKE THE BRUTE FORCE EXAMPLE IN CLASS WITH THE FOODS...we don't ACTUALLY build/store the search tree,
    # we just update our current optimal solution!

    # Now just construct solution as a list of lists of cow names and return it
    brute_sol = []
    for trip in an_optimal_solution:
        trip_names = []
        for cow in trip:
            trip_names.append(cow[0])
        brute_sol.append(trip_names)
    return brute_sol

# Problem 4
def compare_cow_transport_algorithms():
    """
    Using the data from ps1_cow_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here. Use the
    default weight limits of 10 for both greedy_cow_transport and
    brute_force_cow_transport.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    # Load in cow data
    cows = load_cows("ps1_cow_data.txt")

    # Now to measure and compare the time it takes to run both algorithms

    # Greedy Cow Transportation
    # starts the clock on keeps track of when it started
    start_greedy = time.time()
    sol_greedy = []
    for i in range(65000):
        sol_greedy = (greedy_cow_transport(cows, limit=10))
    # stops the clock and keeps track when it ended
    end_greedy = time.time()
    print("Greedy Cow Transportation Result")
    print("A potential optimal solution is " + str(sol_greedy) + "  -> " + str(len(sol_greedy)) + " trips")
    print("Time to carry out is " + str(end_greedy - start_greedy) + " seconds.", end="\n\n")

    # Brute Force Cow Transportation
    start_brute = time.time()
    sol_brute = brute_force_cow_transport(cows, limit=10)
    end_brute = time.time()
    print("Brute Force Cow Transportation Result")
    print("An optimal solution is " + str(sol_brute) + "  -> " + str(len(sol_brute)) + " trips")
    print("Time to carry out is " + str(end_brute - start_brute) + " seconds.", end="\n\n")


def main():
    # Test greedy and brute force algorithms and compare their time efficiency (measure how long they take to run).
    compare_cow_transport_algorithms()


main()
