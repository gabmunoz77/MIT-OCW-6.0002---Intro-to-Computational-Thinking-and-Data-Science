###########################
# 6.0002 Problem Set 1b: Space Change
# Name: Gabriel Munoz
# Collaborators: None*
# Time: Wednesday, May 19, 2021 - Tuesday, June 1, 2021
# Author: charz, cdenise

#================================
# Part B: Golden Eggs
#================================

# FULL DISCLOSURE
# I DO NOT TAKE CREDIT FOR THE SOLUTION HERE IN "ps1b.py" AS MY OWN.
# I struggled for long hours on this part of the assignment, ps1b, and after a long time of getting
# nowhere, I looked to the internet for help and solutions so that I could learn what I needed to and move on.
# I decided to study how other people had approached this problem and what I was doing wrong in my work and reasoning.
# Reading through "Snowball-Wang"'s work in particular made the most sense to me and was close to my working solution.
# After reviewing it, I found it was not quite correct, and decided to debug it.
# What I have in this file is what I came up with.

# *Referenced:
# For DP algorithm --> it was not quite correct, so I made the few changes to make it a correct working solution
# https://github.com/Snowball-Wang/MIT_6.0002_Introduction_to_Computational_Thinking_and_Data_Science/blob/master/ps1/ps1b.py
# For the function to build a random tuple solely for purposes of testing the solution to ps1b in this file
# https://github.com/dorond/MIT-6.0002/blob/master/PS1/ps1b.py

import random

# Problem 1
def dp_make_weight(egg_weights, target_weight, memo = {}):
    """
    Find number of eggs to bring back, using the smallest number of eggs. Assumes there is
    an infinite supply of eggs of each weight, and there is always a egg of value 1.
    
    Parameters:
    egg_weights - tuple of integers, available egg weights sorted from smallest to largest value (1 = d1 < d2 < ... < dk)
    target_weight - int, amount of weight we want to find eggs to fit
    memo - dictionary, OPTIONAL parameter for memoization (you may not need to use this parameter depending on your implementation)
    
    Returns: int, smallest number of eggs needed to make target weight
    """
    # Snowball Wang - github
    # Here I detail exactly what I changed from Wang's solution to debug it and get it running correctly.
    min_nums = target_weight
    # CHANGE: if target_weight <= 0: -> if target_weight == 0:
    # Don't need to check whether target_weight is less than 0 since we already ensure target_weight is never negative
    #   in the list comprehension (for weight in) [k for k in egg_weights if k <= target_weight]
    if target_weight == 0:
        # CHANGE: return 1 -> return 0
        # Should return 0 because when you reach a 0 weight capacity, you will not take any eggs
        # Counterexample to return 1 -> egg_weights = (1, 6, 10), n = 12 would return min_eggs = 3 instead of 2
        # Leaving return 1 wrongly stored min_nums for certain values (6:2 instead of 6:1) by adding extra 1 to num_eggs
        return 0
    # MY COMMENT: Can add a couple of edge cases where we know solution without needing to proceed to recursion
    elif target_weight in egg_weights:
        return 1
    elif len(egg_weights) == 1:
        return target_weight
    # Look up memo to check out if there is already a best solution
    elif target_weight in memo:
        return memo[target_weight]
    else:
        # MY COMMENT: list comprehension (k <= target_weight) to compute valid values for egg weights to take
        # (can't take weight 6 when remaining weight is 5 for example, can only take eggs of weight 1)
        for weight in [k for k in egg_weights if k <= target_weight]:
            # Divide the problem into several sub-problems
            # MY COMMENT: adding 1 to dp_make_weight(...) because when you recur you're taking an egg, so the addition
            #   ensures we're adding up the eggs when we return from recursive call(s)
            num_eggs = 1 + dp_make_weight(egg_weights, target_weight - weight, memo)
            if num_eggs < min_nums:
                min_nums = num_eggs
            memo[target_weight] = min_nums
    return min_nums

# dorond - github
def buildRandomEggTuple(numItems, maxWeight):
    egg_weights = []
    for i in range(numItems):
        egg_weights.append(random.randint(1, maxWeight))
    return tuple(sorted(egg_weights))


# EXAMPLE TESTING CODE, feel free to add more if you'd like
if __name__ == '__main__':
    egg_weights = (1, 5, 10, 25)
    n = 99
    print("Egg weights = " + str(egg_weights))
    print("n = " + str(n))
    print("Expected ouput: 9 (3 * 25 + 2 * 10 + 4 * 1 = 99)")
    print("Actual output:", dp_make_weight(tuple(sorted(egg_weights, reverse=True)), n, memo={}))
    print()

    # Additional Test cases
    egg_weights = (1, 5, 10, 25)
    n = 16
    print("Egg weights = " + str(egg_weights))
    print("n = " + str(n))
    print("Expected ouput: 3 (1 * 10 + 1 * 5 + 1 * 1 = 16)")
    print("Actual output:", dp_make_weight(tuple(sorted(egg_weights, reverse=True)), n, memo={}))
    print()

    egg_weights = (1, 9, 90, 91)
    n = 99
    print("Egg weights = " + str(egg_weights))
    print("n = " + str(n))
    print("Expected ouput: 2 (1 * 90 + 1 * 9 = 99)")
    print("Actual output:", dp_make_weight(tuple(sorted(egg_weights, reverse=True)), n, memo={}))
    print()

    egg_weights = (1, 4, 9, 89, 90, 91)
    n = 93
    print("Egg weights = " + str(egg_weights))
    print("n = " + str(n))
    print("Expected ouput: 2 (1 * 89 + 1 * 4 = 93)")
    print("Actual output:", dp_make_weight(tuple(sorted(egg_weights, reverse=True)), n, memo={}))
    print()

    egg_weights = (1, 6, 9)
    n = 14
    print("Egg weights = " + str(egg_weights))
    print("n = " + str(n))
    print("Expected ouput: 4 (2 * 6 + 2 * 2 = 14)")
    print("Actual output:", dp_make_weight(tuple(sorted(egg_weights, reverse=True)), n, memo={}))
    print()

    egg_weights = (1, 6, 10)
    n = 12
    print("Egg weights = " + str(egg_weights))
    print("n = " + str(n))
    print("Expected ouput: 2 (2 * 6 = 12)")
    print("Actual output:", dp_make_weight(tuple(sorted(egg_weights, reverse=True)), n, memo={}))
    print()

    egg_weights = buildRandomEggTuple(10, 20)
    n = 99
    print("Egg weights = " + str(egg_weights))
    print("n = " + str(n))
    print("Expected ouput: ???")
    print("Actual output:", dp_make_weight(egg_weights, n))
    print()
