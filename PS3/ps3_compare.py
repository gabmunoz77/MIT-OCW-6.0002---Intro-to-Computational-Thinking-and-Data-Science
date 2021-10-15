# -*- coding: utf-8 -*-
# Problem Set 3: Simulating robots
# Name: Gabriel Munoz
# Collaborators (discussion): None
# Time: Friday, July 9, 2021 - Thursday, July 16, 2021

# READ: Wrote this to compare ps3 solutions--one using a list of lists for the Room class and one as I originally
#       intended, using a dictionary. For this assignment, there appears to be no significant difference in their
#       performance time-wise. Using a dictionary is the more natural and simple way to write the code though.

import ps3
import ps3_dict
import math
import pylab

num_robots = 2
speed = 1.0
capacity = 1
room_sizes = [16, 36, 64, 100, 144, 196, 256, 324, 400, 484, 576, 676, 784, 900]
dirt_amount = 3
min_coverage = 0.75
num_trials = 20
robot_type_list = ps3.StandardRobot
robot_type_dict = ps3_dict.StandardRobot
times_list = []
times_dict = []
for size in room_sizes:
    times_list.append(ps3.run_simulation(num_robots, speed, capacity, math.sqrt(size), math.sqrt(size), dirt_amount,
                                         min_coverage, num_trials, robot_type_list))
    times_dict.append(ps3_dict.run_simulation(num_robots, speed, capacity, math.sqrt(size), math.sqrt(size),
                                              dirt_amount, min_coverage, num_trials, robot_type_dict))

pylab.plot(room_sizes, times_list)
pylab.plot(room_sizes, times_dict)
pylab.title("Comparing List and Dictionary Room Implementations by Time Efficiency")
pylab.legend(("List", "Dictionary"))
pylab.xlabel("Room size (width x height")
pylab.ylabel("Time/Steps")
pylab.show()

