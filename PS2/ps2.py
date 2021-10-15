# 6.0002 Problem Set 5
# Graph optimization
# Name: Gabriel Munoz
# Collaborators: None
# Time: Wednesday, June 9, 2021 - Wednesday, June 30, 2021

#
# Finding shortest paths through MIT buildings
#
import unittest
from graph import Digraph, Node, WeightedEdge

#
# Problem 2: Building up the Campus Map
#
# Problem 2a: Designing your graph
#
# What do the graph's nodes represent in this problem? What
# do the graph's edges represent? Where are the distances
# represented?
#
# Answer:
#   In this problem, the graph's nodes represent the different buildings on the MIT campus. The edges represent the
# the paths connecting one building to another. The distances between the buildings are represented by the weights of
# the edges in the graph.
#


# Problem 2b: Implementing load_map
def load_map(map_filename):
    """
    Parses the map file and constructs a directed graph

    Parameters:
        map_filename : name of the map file

    Assumes:
        Each entry in the map file consists of the following four positive
        integers, separated by a blank space:
            From To TotalDistance DistanceOutdoors
        e.g.
            32 76 54 23
        This entry would become an edge from 32 to 76.

    Returns:
        a Digraph representing the map
    """

    # Initialize a directed graph
    dig = Digraph()

    # Open the file for reading
    print("Loading map from file...")
    map_file = open(map_filename, 'r')
    # each line we read from the file represents a Weighted Edge (source, destination, totalDistance, distanceOutdoors)
    for line in map_file:
        # strip away leading and trailing whitespace and split along spaces
        line = line.strip().split()
        # for each edge, check whether parent and child Nodes exist--if they don't, create and add them to digraph
        parent = Node(line[0])
        if not dig.has_node(parent):
            dig.add_node(parent)
        child = Node(line[1])
        if not dig.has_node(child):
            dig.add_node(child)
        # now create Weighted Edge and add it to digraph
        dig.add_edge(WeightedEdge(parent, child, int(line[2]), int(line[3])))
    # close the file when done to avoid corruption and return digraph
    map_file.close()
    return dig


# Problem 2c: Testing load_map
# Include the lines used to test load_map below, but comment them out
# graph = load_map("test_load_map.txt")
# print(graph)
# print(Node("a"))


#
# Problem 3: Finding the Shortest Path using Optimized Search Method
#
# Problem 3a: Objective function
#
# What is the objective function for this problem? What are the constraints?
#
# Answer:
#   The objective function in this problem is the total distance traveled--we are trying to minimize the total distance
# traveled on any given path on the MIT campus. The constraint is the distance traveled outdoors.
#

# Problem 3b: Implement get_best_path


# From textbook John V. Guttag Chapter 12, p.197
def printPath(path):
    """Assumes path is a list of nodes"""
    result = ''
    for i in range(len(path)):
        result = result + str(path[i])
        if i != len(path) - 1:
            result = result + '->'
    return result


def get_best_path(digraph, start, end, path, max_dist_outdoors, best_dist,
                  best_path):
    """
    Finds the shortest path between buildings subject to constraints.

    Parameters:
        digraph: Digraph instance
            The graph on which to carry out the search
        start: string
            Building number at which to start
        end: string
            Building number at which to end
        path: list composed of [[list of strings], int, int]
            Represents the current path of nodes being traversed. Contains
            a list of node names, total distance traveled, and total
            distance outdoors.
        max_dist_outdoors: int
            Maximum distance spent outdoors on a path
        best_dist: int
            The smallest distance between the original start and end node
            for the initial problem that you are trying to solve
        best_path: list of strings
            The shortest path found so far between the original start
            and end node.

    Returns:
        A tuple with the shortest-path from start to end, represented by
        a list of building numbers (in strings), [n_1, n_2, ..., n_k],
        where there exists an edge from n_i to n_(i+1) in digraph,
        for all 1 <= i < k and the distance of that path.

        If there exists no path that satisfies max_total_dist and
        max_dist_outdoors constraints, then return None.
    """

    # Nodes start and end
    start = Node(start)
    end = Node(end)

    # Construct path and display it
    path = [path[0] + [start], path[1], path[2]]
    print("Current DFS path: " + printPath(path[0]))

    # Check that start and end are valid Nodes
    if not (digraph.has_node(start) and digraph.has_node(end)):
        raise ValueError("Node(s) not in graph.")
    # Check whether start and end are the same Node--if so, we can return
    elif start == end:
        return path[0], path[1]
    # Continue on to construct paths through graph
    else:
        # iterate through every edge of current start Node
        for edge in digraph.get_edges_for_node(start):
            # check whether we've visited the destination Node already (avoiding cycles)
            if edge.get_destination() not in path[0]:
                # take the distance traveled and distance traveled outdoors and check if path stays within constraints
                dist_added = edge.get_total_distance()
                dist_outdoors_added = edge.get_outdoor_distance()
                # Only want to continue if we don't exceed the max distance traveled/outdoors
                if path[1] + dist_added <= best_dist and path[2] + dist_outdoors_added <= max_dist_outdoors:
                    # if we stay within constraints, create an updated potential/test path
                    # -> note: don't want to "commit" to adding edge weights when we might backtrack--updating path
                    #           directly would persist through recursion and back to an outer frame
                    potential_path = [path[0], path[1] + dist_added, path[2] + dist_outdoors_added]
                    # recursion with destination Node as new start
                    new_path, new_dist = get_best_path(digraph, edge.get_destination(), end, potential_path,
                                                       max_dist_outdoors, best_dist, best_path)
                    # if we found a new shortest path, update best_path and best_dist
                    if new_path is not None:
                        best_path = new_path
                        best_dist = new_dist
                # if visiting current Node destination would break constraints, let user know
                else:
                    print("Visiting " + str(edge.get_destination()) + " would break the constraints.")
            # if Node found has already been visited, print out a message
            else:
                print("Already visited " + str(edge.get_destination()))
    # once finished, we can return the best path and distance
    return best_path, best_dist


# Test function for get_best_path() with smaller map test text file
def test_get_best_path(start, end):
    print("Testing load_map...")
    graph = load_map("test_load_map.txt")
    print(graph, end="\n\n")

    path = [[], 0, 0]
    max_dist_outdoors = 22
    best_dist = 10000
    best_path = None
    shortest_path, shortest_dist = get_best_path(graph, start, end, path, max_dist_outdoors,
                                                 best_dist, best_path)
    print()

    if shortest_path is None:
        print("There is no path from " + start + " to " + end + ".\n")
    elif len(shortest_path) == 1:
        print("We are already here!")
    else:
        print("The shortest path from " + start + " to " + end + " is " + printPath(shortest_path), end="\n")
        print("The shortest path has a distance of " + str(shortest_dist) + " meters.\n")
    print()


# Lets test get_best_path--("a", "h"), ("a", "i"), ("b", "h"), ("b", "i"), etc.
test_get_best_path("a", "h")


# Problem 3c: Implement directed_dfs
def directed_dfs(digraph, start, end, max_total_dist, max_dist_outdoors):
    """
    Finds the shortest path from start to end using a directed depth-first
    search. The total distance traveled on the path must not
    exceed max_total_dist, and the distance spent outdoors on this path must
    not exceed max_dist_outdoors.

    Parameters:
        digraph: Digraph instance
            The graph on which to carry out the search
        start: string
            Building number at which to start
        end: string
            Building number at which to end
        max_total_dist: int
            Maximum total distance on a path
        max_dist_outdoors: int
            Maximum distance spent outdoors on a path

    Returns:
        The shortest-path from start to end, represented by
        a list of building numbers (in strings), [n_1, n_2, ..., n_k],
        where there exists an edge from n_i to n_(i+1) in digraph,
        for all 1 <= i < k

        If there exists no path that satisfies max_total_dist and
        max_dist_outdoors constraints, then raises a ValueError.
    """

    # Initialize variables needed and call get_best_path()
    path = [[], 0, 0]
    best_dist = 99999
    best_path = None
    shortest_path, shortest_dist = get_best_path(digraph, start, end, path, max_dist_outdoors, best_dist, best_path)

    # Return the shortest path or raise an appropriate error if needed
    if shortest_path is None:
        raise ValueError("There is no path from start to end.")
    if shortest_dist > max_total_dist:
        raise ValueError("Shortest path found exceeds max total distance traveled allowed.")
    else:
        # need to return a path of strings--list comprehension makes this easy
        return [node.get_name() for node in shortest_path]


# ================================================================
# Begin tests -- you do not need to modify anything below this line
# ================================================================

class Ps2Test(unittest.TestCase):
    LARGE_DIST = 99999

    def setUp(self):
        self.graph = load_map("mit_map.txt")

    def test_load_map_basic(self):
        self.assertTrue(isinstance(self.graph, Digraph))
        self.assertEqual(len(self.graph.nodes), 37)
        all_edges = []
        for _, edges in self.graph.edges.items():
            all_edges += edges  # edges must be dict of node -> list of edges
        all_edges = set(all_edges)
        self.assertEqual(len(all_edges), 129)

    def _print_path_description(self, start, end, total_dist, outdoor_dist):
        constraint = ""
        if outdoor_dist != Ps2Test.LARGE_DIST:
            constraint = "without walking more than {}m outdoors".format(
                outdoor_dist)
        if total_dist != Ps2Test.LARGE_DIST:
            if constraint:
                constraint += ' or {}m total'.format(total_dist)
            else:
                constraint = "without walking more than {}m total".format(
                    total_dist)

        print("------------------------")
        print("Shortest path from Building {} to {} {}".format(
            start, end, constraint))

    def _test_path(self,
                   expectedPath,
                   total_dist=LARGE_DIST,
                   outdoor_dist=LARGE_DIST):
        start, end = expectedPath[0], expectedPath[-1]
        self._print_path_description(start, end, total_dist, outdoor_dist)
        dfsPath = directed_dfs(self.graph, start, end, total_dist, outdoor_dist)
        print("Expected: ", expectedPath)
        print("DFS: ", dfsPath)
        self.assertEqual(expectedPath, dfsPath)

    def _test_impossible_path(self,
                              start,
                              end,
                              total_dist=LARGE_DIST,
                              outdoor_dist=LARGE_DIST):
        self._print_path_description(start, end, total_dist, outdoor_dist)
        with self.assertRaises(ValueError):
            directed_dfs(self.graph, start, end, total_dist, outdoor_dist)

    def test_path_one_step(self):
        self._test_path(expectedPath=['32', '56'])

    def test_path_no_outdoors(self):
        self._test_path(
            expectedPath=['32', '36', '26', '16', '56'], outdoor_dist=0)

    def test_path_multi_step(self):
        self._test_path(expectedPath=['2', '3', '7', '9'])

    def test_path_multi_step_no_outdoors(self):
        self._test_path(
            expectedPath=['2', '4', '10', '13', '9'], outdoor_dist=0)

    def test_path_multi_step2(self):
        self._test_path(expectedPath=['1', '4', '12', '32'])

    def test_path_multi_step_no_outdoors2(self):
        self._test_path(
            expectedPath=['1', '3', '10', '4', '12', '24', '34', '36', '32'],
            outdoor_dist=0)

    def test_impossible_path1(self):
        self._test_impossible_path('8', '50', outdoor_dist=0)

    def test_impossible_path2(self):
        self._test_impossible_path('10', '32', total_dist=100)


if __name__ == "__main__":
    unittest.main()
