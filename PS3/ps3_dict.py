# -*- coding: utf-8 -*-
# Problem Set 3: Simulating robots
# Name: Gabriel Munoz
# Collaborators (discussion): None
# Time: Friday, July 9, 2021 - Friday, July 16, 2021

# ** 07/15/2021 - Try this assignment out switching the list of lists for the room data structure for a dictionary,
# and test the efficiency differences...

# ** 07/16/2021 - See ps3_compare.py, ps3_dict.py, and ps3_dict_tests_f16.py
# ** --> note: There appear to be no significant time efficiency differences between a List and Dictionary
#               implementation for Room. One is faster than the other at different points in multiple trials with
#               different inputs, etc. Using a dictionary is ever so slightly more natural and easy to work with, but
#               it's really negligible. My initial instinct and reasoning for using a dictionary was right...
#               Trust yourself!


import math
import random

import ps3_visualize
import pylab

# For python 2.7:
from ps3_verify_movement27 import test_robot_movement


# === Provided class Position
class Position(object):
    """
    A Position represents a location in a two-dimensional room, where
    coordinates are given by floats (x, y).
    """
    def __init__(self, x, y):
        """
        Initializes a position with coordinates (x, y).
        """
        self.x = x
        self.y = y
        
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y
    
    def get_new_position(self, angle, speed):
        """
        Computes and returns the new Position after a single clock-tick has
        passed, with this object as the current position, and with the
        specified angle and speed.

        Does NOT test whether the returned position fits inside the room.

        angle: float representing angle in degrees, 0 <= angle < 360
        speed: positive float representing speed

        Returns: a Position object representing the new position.
        """
        old_x, old_y = self.get_x(), self.get_y()
        
        # Compute the change in position
        delta_y = speed * math.cos(math.radians(angle))
        delta_x = speed * math.sin(math.radians(angle))
        
        # Add that to the existing position
        new_x = old_x + delta_x
        new_y = old_y + delta_y
        
        return Position(new_x, new_y)

    def __str__(self):  
        return "Position: " + str(math.floor(self.x)) + ", " + str(math.floor(self.y))


# === Problem 1
class RectangularRoom(object):
    """
    A RectangularRoom represents a rectangular region containing clean or dirty
    tiles.

    A room has a width and a height and contains (width * height) tiles. Each tile
    has some fixed amount of dirt. The tile is considered clean only when the amount
    of dirt on this tile is 0.
    """
    def __init__(self, width, height, dirt_amount):
        """
        Initializes a rectangular room with the specified width, height, and 
        dirt_amount on each tile.

        width: an integer > 0
        height: an integer > 0
        dirt_amount: an integer >= 0
        """
        # Will use a list of lists (width x height, 2d array) to represent the room, with indices being (x, y) pairs,
        # the positions representing the tiles, and the values stored the amount of dirt left on each tile
        # --> note: Would a dictionary of (x, y) keys mapped to tile dirt amount values be better? More efficient?
        #           Hmmm, accessing a list vs a dictionary is both O(1) on average (but dict is O(n) at worst case)
        #           Dictionary more practical, and for this assignment, probably as efficient as list of lists...
        self.width = int(width)
        self.height = int(height)
        # Force height and width to be ints here for list comprehension to work (show_plot_room_shape() uses floats)
        #self.room = [[dirt_amount for j in range(self.height)] for i in range(self.width)]
        self.room = {(i, j): dirt_amount for i in range(self.width) for j in range(self.height)}

    def clean_tile_at_position(self, pos, capacity):
        """
        Mark the tile under the position pos as cleaned by capacity amount of dirt.

        Assumes that pos represents a valid position inside this room.

        pos: a Position object
        capacity: the amount of dirt to be cleaned in a single time-step
                  can be negative which would mean adding dirt to the tile

        Note: The amount of dirt on each tile should be NON-NEGATIVE.
              If the capacity exceeds the amount of dirt on the tile, mark it as 0.
        """
        # Need the x and y coordinates of Position pos rounded down to determine what tile robot is on
        tile_x, tile_y = math.floor(pos.get_x()), math.floor(pos.get_y())

        # Clean tile by capacity amount if it is dirty (dirt amount on tile > 0)
        if self.room[tile_x, tile_y] > 0:
            self.room[tile_x, tile_y] -= capacity

        # If dirt is negative (capacity was > dirt left, or for some reason was already negative), mark as 0 (clean)
        if self.room[tile_x, tile_y] < 0:
            self.room[tile_x, tile_y] = 0

    def is_tile_cleaned(self, m, n):
        """
        Return True if the tile (m, n) has been cleaned.

        Assumes that (m, n) represents a valid tile inside the room.

        m: an integer
        n: an integer
        
        Returns: True if the tile (m, n) is cleaned, False otherwise

        Note: The tile is considered clean only when the amount of dirt on this
              tile is 0.
        """
        # If room tile value (dirt amount) at (m, n) equals 0, it is cleaned, and will return True, False otherwise
        return self.room[m, n] == 0

    def get_num_cleaned_tiles(self):
        """
        Returns: an integer; the total number of clean tiles in the room
        """
        # --> note: Is there a more efficient way to do this without numpy arrays or pandas dataframes?
        # Should this make me go with the dictionary approach for representing Rooms?
        # num_cleaned_tiles = 0
        # for i in range(self.width):
        #     for j in range(self.height):
        #         if self.room[i][j] == 0:
        #             num_cleaned_tiles += 1
        # return num_cleaned_tiles
        #return sum([1 for j in range(self.height) for i in range(self.width) if self.room[i, j] == 0])
        return sum([1 for k, v in self.room.items() if self.room[k] == 0])
        
    def is_position_in_room(self, pos):
        """
        Determines if pos is inside the room.

        pos: a Position object.
        Returns: True if pos is in the room, False otherwise.
        """
        # Need to check if 0 <= x < w and 0 <= y < h are true
        return (0 <= pos.get_x() < self.width) and (0 <= pos.get_y() < self.height)
        
    def get_dirt_amount(self, m, n):
        """
        Return the amount of dirt on the tile (m, n)
        
        Assumes that (m, n) represents a valid tile inside the room.

        m: an integer
        n: an integer

        Returns: an integer
        """
        # Just need to index Room room with (m, n) assuming they are valid integers representing a tile
        return self.room[m, n]
        
    def get_num_tiles(self):
        """
        Returns: an integer; the total number of tiles in the room
        """
        # do not change -- implement in subclasses.
        raise NotImplementedError 
        
    def is_position_valid(self, pos):
        """
        pos: a Position object.
        
        returns: True if pos is in the room and (in the case of FurnishedRoom) 
                 if position is unfurnished, False otherwise.
        """
        # do not change -- implement in subclasses
        raise NotImplementedError         

    def get_random_position(self):
        """
        Returns: a Position object; a random position inside the room
        """
        # do not change -- implement in subclasses
        raise NotImplementedError        


class Robot(object):
    """
    Represents a robot cleaning a particular room.

    At all times, the robot has a particular position and direction in the room.
    The robot also has a fixed speed and a fixed cleaning capacity.

    Subclasses of Robot should provide movement strategies by implementing
    update_position_and_clean, which simulates a single time-step.
    """
    def __init__(self, room, speed, capacity):
        """
        Initializes a Robot with the given speed and given cleaning capacity in the 
        specified room. The robot initially has a random direction and a random 
        position in the room.

        room:  a RectangularRoom object.
        speed: a float (speed > 0)
        capacity: a positive interger; the amount of dirt cleaned by the robot 
                  in a single time-step
        """
        # Set the Robot attributes to the room, speed, and capacity parameters
        self.room = room
        self.speed = speed
        self.capacity = capacity
        # First the seed (starting it here should give us the same numbers when rerunning and make debugging easier)
        # --> note: when actually running to test functionality and actual use, DO NOT USE
        #random.seed(0)
        # How should we assign the random Position and direction to Robot? get.random_position() not implemented...
        # --> note: We will just call get_random_position() for an EmptyRoom--will implement that first
        self.position = room.get_random_position()
        # direction is a 0 <= float < 360 angle measured from north
        # --> note: also could use random.uniform(0, 360), with a very subtle difference...
        #           random.random() is a closed interval on the left, but open on the right, while random.uniform() is
        #           a closed interval on both ends...
        #self.direction = random.uniform(0, 360)
        self.direction = 360*random.random()

    def get_robot_position(self):
        """
        Returns: a Position object giving the robot's position in the room.
        """
        return self.position

    def get_robot_direction(self):
        """
        Returns: a float d giving the direction of the robot as an angle in
        degrees, 0.0 <= d < 360.0.
        """
        return self.direction

    def set_robot_position(self, position):
        """
        Set the position of the robot to position.

        position: a Position object.
        """
        self.position = position

    def set_robot_direction(self, direction):
        """
        Set the direction of the robot to direction.

        direction: float representing an angle in degrees
        """
        self.direction = direction

    def update_position_and_clean(self):
        """
        Simulate the raise passage of a single time-step.

        Move the robot to a new random position (if the new position is invalid, 
        rotate once to a random new direction, and stay stationary) and mark the tile it is on as having
        been cleaned by capacity amount. 
        """
        # do not change -- implement in subclasses
        raise NotImplementedError

# === Problem 2
class EmptyRoom(RectangularRoom):
    """
    An EmptyRoom represents a RectangularRoom with no furniture.
    """
    def get_num_tiles(self):
        """
        Returns: an integer; the total number of tiles in the room
        """
        # The total number of tiles in the Room is its width x height (or the sum of # of rows & cols in the Room array)
        return int(self.width*self.height)
        
    def is_position_valid(self, pos):
        """
        pos: a Position object.
        
        Returns: True if pos is in the room, False otherwise.
        """
        # For an unfurnished room, a position is valid if it is inside the Room
        return self.is_position_in_room(pos)
        
    def get_random_position(self):
        """
        Returns: a Position object; a valid random position (inside the room).
        """
        # Will use this by default to initialize a Robot's position as a random Position inside the Room
        # Need to access Room's width and height to guarantee random Position is inside the Room
        #rand_x = random.uniform(0, self.width)
        #rand_y = random.uniform(0, self.height)
        rand_x = self.width*random.random()
        rand_y = self.height*random.random()
        return Position(rand_x, rand_y)

class FurnishedRoom(RectangularRoom):
    """
    A FurnishedRoom represents a RectangularRoom with a rectangular piece of 
    furniture. The robot should not be able to land on these furniture tiles.
    """
    def __init__(self, width, height, dirt_amount):
        """ 
        Initializes a FurnishedRoom, a subclass of RectangularRoom. FurnishedRoom
        also has a list of tiles which are furnished (furniture_tiles).
        """
        # This __init__ method is implemented for you -- do not change.
        
        # Call the __init__ method for the parent class
        RectangularRoom.__init__(self, width, height, dirt_amount)
        # Adds the data structure to contain the list of furnished tiles
        self.furniture_tiles = []
        
    def add_furniture_to_room(self):
        """
        Add a rectangular piece of furniture to the room. Furnished tiles are stored 
        as (x, y) tuples in the list furniture_tiles 
        
        Furniture location and size is randomly selected. Width and height are selected
        so that the piece of furniture fits within the room and does not occupy the 
        entire room. Position is selected by randomly selecting the location of the 
        bottom left corner of the piece of furniture so that the entire piece of 
        furniture lies in the room.
        """
        # This addFurnitureToRoom method is implemented for you. Do not change it.
        furniture_width = random.randint(1, self.width - 1)
        furniture_height = random.randint(1, self.height - 1)

        # Randomly choose bottom left corner of the furniture item.    
        f_bottom_left_x = random.randint(0, self.width - furniture_width)
        f_bottom_left_y = random.randint(0, self.height - furniture_height)

        # Fill list with tuples of furniture tiles.
        for i in range(f_bottom_left_x, f_bottom_left_x + furniture_width):
            for j in range(f_bottom_left_y, f_bottom_left_y + furniture_height):
                self.furniture_tiles.append((i,j))             

    def is_tile_furnished(self, m, n):
        """
        Return True if tile (m, n) is furnished.
        """
        return (m, n) in self.furniture_tiles
        
    def is_position_furnished(self, pos):
        """
        pos: a Position object.

        Returns True if pos is furnished and False otherwise
        """
        # Position x and y coordinates are floats, so just need to check whether their floors are furnished or not
        return (math.floor(pos.get_x()), math.floor(pos.get_y())) in self.furniture_tiles
        
    def is_position_valid(self, pos):
        """
        pos: a Position object.
        
        returns: True if pos is in the room and is unfurnished, False otherwise.
        """
        return self.is_position_in_room(pos) and not self.is_position_furnished(pos)
        
    def get_num_tiles(self):
        """
        Returns: an integer; the total number of tiles in the room that can be accessed.
        """
        # The number of tiles that can be accessed = number of total tiles in the room - number of tiles furnished
        return self.width*self.height - sum([1 for j in range(self.height) for i in range(self.width)
                                             if (i, j) in self.furniture_tiles])
        
    def get_random_position(self):
        """
        Returns: a Position object; a valid random position (inside the room and not in a furnished area).
        """
        # Simplest way to do this--maybe not the best or most efficient way--is to generate random Positions until
        # one is valid (not in a furnished area)
        # --> note: There must be a way to break up the space of possible options for the random floats to be generated
        # from--i.e. we would need to find the intervals of valid numbers and choose from the aggregate of those...hmmm
        # ...might be more efficient to just do this loop...(for the above we'd need to iterate through all tiles, skip
        # over every furnished tile, and capture the endpoints of every interval as tuples in another list...
        # then generate random floats over an interval of equivalent length as the aggregate interval...and then shift
        # the number generated to the original (non-aggregated) interval...e.g. for 0 < n < 0.2 and 0.8 < n 1.0, do
        # 0 < n < 0.4 and shift by the distance of the invalid region...0.1 would actually be 0.1 + 0.6 = 0.7, etc...
        # could generalize this for more than 2 intervals...write helper function for it?
        # --> NOTE: the unit test for this function only takes ~1 second...
        while True:
            rand_x = self.width * random.random()
            rand_y = self.height * random.random()
            potential_pos = Position(rand_x, rand_y)
            if self.is_position_valid(potential_pos):
                return potential_pos

# === Problem 3
class StandardRobot(Robot):
    """
    A StandardRobot is a Robot with the standard movement strategy.

    At each time-step, a StandardRobot attempts to move in its current
    direction; when it would hit a wall or furtniture, it *instead*
    chooses a new direction randomly.
    """
    def update_position_and_clean(self):
        """
        Simulate the raise passage of a single time-step.

        Move the robot to a new random position (if the new position is invalid, 
        rotate once to a random new direction, and stay stationary) and clean the dirt on the tile
        by its given capacity. 
        """
        # Calculate new position
        curr_pos = self.get_robot_position()
        new_pos = curr_pos.get_new_position(self.get_robot_direction(), self.speed)
        # If new position is valid, move there, and clean the tile
        if self.room.is_position_valid(new_pos):
            self.set_robot_position(new_pos)
            self.room.clean_tile_at_position(new_pos, self.capacity)
        # Otherwise, rotate the robot to a random new direction, DON'T move, and DON'T clean
        else:
            self.set_robot_direction(360*random.random())


# Uncomment this line to see your implementation of StandardRobot in action!
#test_robot_movement(StandardRobot, EmptyRoom)
#test_robot_movement(StandardRobot, FurnishedRoom)

# === Problem 4
class FaultyRobot(Robot):
    """
    A FaultyRobot is a robot that will not clean the tile it moves to and
    pick a new, random direction for itself with probability p rather
    than simply cleaning the tile it moves to.
    """
    p = 0.15

    @staticmethod
    def set_faulty_probability(prob):
        """
        Sets the probability of getting faulty equal to PROB.

        prob: a float (0 <= prob <= 1)
        """
        FaultyRobot.p = prob
    
    def gets_faulty(self):
        """
        Answers the question: Does this FaultyRobot get faulty at this timestep?
        A FaultyRobot gets faulty with probability p.

        returns: True if the FaultyRobot gets faulty, False otherwise.
        """
        return random.random() < FaultyRobot.p
    
    def update_position_and_clean(self):
        """
        Simulate the passage of a single time-step.

        Check if the robot gets faulty. If the robot gets faulty,
        do not clean the current tile and change its direction randomly.

        If the robot does not get faulty, the robot should behave like
        StandardRobot at this time-step (checking if it can move to a new position,
        move there if it can, pick a new direction and stay stationary if it can't)
        """
        # If Robot is faulty at current time-step, don't clean tile, just change it's direction to a new random one
        if self.gets_faulty():
            self.set_robot_direction(360*random.random())
        # Otherwise, works like StandardRobot--clean next tile if it can move to it, or pick a random new direction
        else:
            # calculate new position
            new_pos = self.get_robot_position().get_new_position(self.get_robot_direction(), self.speed)
            # if new position is valid, move to and clean the tile
            if self.room.is_position_valid(new_pos):
                self.set_robot_position(new_pos)
                self.room.clean_tile_at_position(new_pos, self.capacity)
            # otherwise, just rotate robot to a random new direction
            else:
                self.set_robot_direction(360*random.random())
        
    
#test_robot_movement(FaultyRobot, EmptyRoom)

# === Problem 5
def run_simulation(num_robots, speed, capacity, width, height, dirt_amount, min_coverage, num_trials,
                  robot_type):
    """
    Runs num_trials trials of the simulation and returns the mean number of
    time-steps needed to clean the fraction min_coverage of the room.

    The simulation is run with num_robots robots of type robot_type, each       
    with the input speed and capacity in a room of dimensions width x height
    with the dirt dirt_amount on each tile.
    
    num_robots: an int (num_robots > 0)
    speed: a float (speed > 0)
    capacity: an int (capacity >0)
    width: an int (width > 0)
    height: an int (height > 0)
    dirt_amount: an int
    min_coverage: a float (0 <= min_coverage <= 1.0)
    num_trials: an int (num_trials > 0)
    robot_type: class of robot to be instantiated (e.g. StandardRobot or
                FaultyRobot)
    """
    # Keep track of time-steps it takes to clean room--will return average number of time steps to clean room
    num_delta_t = []
    # Simulation runs for num_trials trials
    for i in range(num_trials):
        # Initialize variables per trial: a width*height Room, num_robots Robots of robot_type, time steps
        room = EmptyRoom(width, height, dirt_amount)
        # Shouldn't this line be good enough? See note below.
        #robots = [robot_type(room, speed, capacity) for j in range(num_robots)]
        # --> note: Is it necessary to set the robots' positions to a new random position like this?
        # After all, the Robot class constructor already initializes a Robot's position to a random position using
        # room.get_random_position()...
        robots = []
        for j in range(num_robots):
            robot = robot_type(room, speed, capacity)
            robot.set_robot_position(room.get_random_position())
            robots.append(robot)
        delta_t = 0
        # Before starting the trial, start animation
        #anim = ps3_visualize.RobotVisualization(num_robots, width, height, False, delay=0.01)
        # Clean room with robot(s) until the min_coverage is met, counting the number of time steps in the process
        while float(room.get_num_cleaned_tiles()/room.get_num_tiles()) < min_coverage:
            for robot in robots:
                robot.update_position_and_clean()
            delta_t += 1
            # Update the animation
            #anim.update(room, robots)
        # Once min_coverage met, store time_steps it took to compute average later
        num_delta_t.append(delta_t)
        # After the trial is over, finish the animation
        #anim.done()
    # Once simulation is finished, return the average
    return sum(num_delta_t)/num_trials


# print ('avg time steps: ' + str(run_simulation(1, 1.0, 1, 5, 5, 3, 1.0, 50, StandardRobot)))
# print ('avg time steps: ' + str(run_simulation(1, 1.0, 1, 10, 10, 3, 0.8, 50, StandardRobot)))
# print ('avg time steps: ' + str(run_simulation(1, 1.0, 1, 10, 10, 3, 0.9, 50, StandardRobot)))
# print ('avg time steps: ' + str(run_simulation(1, 1.0, 1, 20, 20, 3, 0.5, 50, StandardRobot)))
# print ('avg time steps: ' + str(run_simulation(3, 1.0, 1, 20, 20, 3, 0.5, 50, StandardRobot)))

# === Problem 6
#
# ANSWER THE FOLLOWING QUESTIONS:
#
# 1)How does the performance of the two robot types compare when cleaning 80%
#       of a 20x20 room?
# --> The relationship between time/steps and number of robots for FaultyRobot is identical to that of StandardRobot,
#       but shifted up by ~500 time steps when there is 1 robot. With 1 robot is when we'd expect to see the greatest
#       effect of having a FaultyRobot--that is, that with probability p = 0.15, the robot will not clean the tile it
#       should and will instead just turn to a random direction. However, we'd expect this effect to be minimized as the
#       the number of robots increases, since if any one robot should fail to clean its tile, there will be many robots
#       already cleaning to "make up" for the faulty robot (expected 8.5 robots to be working properly at any point in
#       time, for 10 total robots). That is indeed what we observe, as the two curves converge as they both appear to
#       asymptotically approach ~200 time steps as the number of robots increases.
#
#
# 2) How does the performance of the two robot types compare when two of each
#       robot cleans 80% of rooms with dimensions 
#       10x30, 20x15, 25x12, and 50x6?
# --> The two Robot types appear to follow the same trend, but the FaultyRobot one is more pronounced at the extremes.
#       This makes sense given the probability p = 0.15 that the robot will not clean the tile it's on and hence either
#       greatly increasing or decreasing the time/steps needed to clear the minimum coverage of tiles cleaned.
#       The general trend observed for both appears to be that the minimum time/steps required for cleaning the room
#       is achieved in rooms with aspect ratios close to 1:1, and that the time/steps increase in both directions as the
#       aspect ratio deviates from 1:1--very long or very wide rectangular rooms take longer to clean than square rooms.
#       I presume this is because for very long or very wide rectangular rooms, you are limiting the directions the
#       robot can turn to find valid tiles to clean (straight up/down or straight left/right). If the room is closer to
#       square one, the robot will have more directions to turn to in order to find valid tiles to clean, i.e. a square
#       room gives a robot a higher probability of finding valid tiles to clean when turning to a random direction than
#       a long or wide rectangular room does, and hence will take fewer time/steps to clean.


def show_plot_compare_strategies(title, x_label, y_label):
    """
    Produces a plot comparing the two robot strategies in a 20x20 room with 80%
    minimum coverage.
    """
    num_robot_range = range(1, 11)
    times1 = []
    times2 = []
    for num_robots in num_robot_range:
        print ("Plotting", num_robots, "robots...")
        times1.append(run_simulation(num_robots, 1.0, 1, 20, 20, 3, 0.8, 20, StandardRobot))
        times2.append(run_simulation(num_robots, 1.0, 1, 20, 20, 3, 0.8, 20, FaultyRobot))
    pylab.plot(num_robot_range, times1)
    pylab.plot(num_robot_range, times2)
    pylab.title(title)
    pylab.legend(('StandardRobot', 'FaultyRobot'))
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    pylab.show()
    
def show_plot_room_shape(title, x_label, y_label):
    """
    Produces a plot showing dependence of cleaning time on room shape.
    """
    aspect_ratios = []
    times1 = []
    times2 = []
    for width in [10, 20, 25, 50]:
        height = 300/width
        print ("Plotting cleaning time for a room of width:", width, "by height:", height)
        aspect_ratios.append(float(width) / height)
        times1.append(run_simulation(2, 1.0, 1, width, height, 3, 0.8, 200, StandardRobot))
        times2.append(run_simulation(2, 1.0, 1, width, height, 3, 0.8, 200, FaultyRobot))
    pylab.plot(aspect_ratios, times1)
    pylab.plot(aspect_ratios, times2)
    pylab.title(title)
    pylab.legend(('StandardRobot', 'FaultyRobot'))
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    pylab.show()


#show_plot_compare_strategies('Time to clean 80% of a 20x20 room, for various numbers of robots','Number of robots','Time / steps')
#show_plot_room_shape('Time to clean 80% of a 300-tile room for various room shapes','Aspect Ratio', 'Time / steps')
