# Problem Set 4: Simulating the Spread of Disease and Bacteria Population Dynamics
# Name: Gabriel Munoz
# Collaborators (Discussion): None
# Time: Tuesday, July 27, 2021 - Tuesday, August 3, 2021

import math
import numpy as np
import pylab as pl
import random

# For reproducible results when dealing with random numbers in debugging
random.seed(0)

##########################
# End helper code
##########################

class NoChildException(Exception):
    """
    NoChildException is raised by the reproduce() method in the SimpleBacteria
    and ResistantBacteria classes to indicate that a bacteria cell does not
    reproduce. You should use NoChildException as is; you do not need to
    modify it or add any code.
    """


def make_one_curve_plot(x_coords, y_coords, x_label, y_label, title):
    """
    Makes a plot of the x coordinates and the y coordinates with the labels
    and title provided.

    Args:
        x_coords (list of floats): x coordinates to graph
        y_coords (list of floats): y coordinates to graph
        x_label (str): label for the x-axis
        y_label (str): label for the y-axis
        title (str): title for the graph
    """
    pl.figure()
    # Added x- and y-limits for readability in the graph
    pl.xlim([0, len(x_coords)])
    pl.ylim([0, 800])
    pl.plot(x_coords, y_coords)
    pl.xlabel(x_label)
    pl.ylabel(y_label)
    pl.title(title)
    pl.show()


def make_two_curve_plot(x_coords,
                        y_coords1,
                        y_coords2,
                        y_name1,
                        y_name2,
                        x_label,
                        y_label,
                        title):
    """
    Makes a plot with two curves on it, based on the x coordinates with each of
    the set of y coordinates provided.

    Args:
        x_coords (list of floats): the x coordinates to graph
        y_coords1 (list of floats): the first set of y coordinates to graph
        y_coords2 (list of floats): the second set of y-coordinates to graph
        y_name1 (str): name describing the first y-coordinates line
        y_name2 (str): name describing the second y-coordinates line
        x_label (str): label for the x-axis
        y_label (str): label for the y-axis
        title (str): the title of the graph
    """
    pl.figure()
    # Added x- and y-limits for readability in the graphs
    pl.xlim([0, len(x_coords)])
    pl.ylim([0, max(y_coords1 + y_coords2) + 10])
    # pl.ylim([0, 900])
    # pl.ylim([0, 180])
    # Added colors and markers for readability in graphs
    pl.plot(x_coords, y_coords1, 'm-', label=y_name1)
    pl.plot(x_coords, y_coords2, 'c-.', label=y_name2)
    pl.legend()
    pl.xlabel(x_label)
    pl.ylabel(y_label)
    pl.title(title)
    pl.show()


##########################
# PROBLEM 1
##########################

class SimpleBacteria(object):
    """A simple bacteria cell with no antibiotic resistance"""

    def __init__(self, birth_prob, death_prob):
        """
        Args:
            birth_prob (float in [0, 1]): Maximum possible reproduction
                probability
            death_prob (float in [0, 1]): Maximum death probability
        """
        self.birth_prob = birth_prob
        self.death_prob = death_prob

    def is_killed(self):
        """
        Stochastically determines whether this bacteria cell is killed in
        the patient's body at a time step, i.e. the bacteria cell dies with
        some probability equal to the death probability each time step.

        Returns:
            bool: True with probability self.death_prob, False otherwise.
        """
        # A float less than or equal to self.death_prob will be chosen self.death_prob * 100 % of the time
        return random.uniform(0.0, 1.0) <= self.death_prob

    def reproduce(self, pop_density):
        """
        Stochastically determines whether this bacteria cell reproduces at a
        time step. Called by the update() method in the Patient and
        TreatedPatient classes.

        The bacteria cell reproduces with probability
        self.birth_prob * (1 - pop_density).

        If this bacteria cell reproduces, then reproduce() creates and returns
        the instance of the offspring SimpleBacteria (which has the same
        birth_prob and death_prob values as its parent).

        Args:
            pop_density (float): The population density, defined as the
                current bacteria population divided by the maximum population

        Returns:
            SimpleBacteria: A new instance representing the offspring of
                this bacteria cell (if the bacteria reproduces). The child
                should have the same birth_prob and death_prob values as
                this bacteria.

        Raises:
            NoChildException if this bacteria cell does not reproduce.
        """
        if random.uniform(0.0, 1.0) <= self.birth_prob * (1 - pop_density):
            return SimpleBacteria(self.birth_prob, self.death_prob)
        else:
            raise NoChildException

    def get_resistant(self):
        """Returns whether the bacteria has antibiotic resistance"""

        # --> note: Will just be an abstract method implemented in ResistantBacteria
        raise NotImplementedError


class Patient(object):
    """
    Representation of a simplified patient. The patient does not take any
    antibiotics and his/her bacteria populations have no antibiotic resistance.
    """
    def __init__(self, bacteria, max_pop):
        """
        Args:
            bacteria (list of SimpleBacteria): The bacteria in the population
            max_pop (int): Maximum possible bacteria population size for
                this patient
        """
        self.bacteria = bacteria
        self.max_pop = max_pop

    def get_total_pop(self):
        """
        Gets the size of the current total bacteria population.

        Returns:
            int: The total bacteria population
        """
        return len(self.bacteria)

    def update(self):
        """
        Update the state of the bacteria population in this patient for a
        single time step. update() should execute the following steps in
        this order:

        1. Determine whether each bacteria cell dies (according to the
           is_killed method) and create a new list of surviving bacteria cells.

        2. Calculate the current population density by dividing the surviving
           bacteria population by the maximum population. This population
           density value is used for the following steps until the next call
           to update()

        3. Based on the population density, determine whether each surviving
           bacteria cell should reproduce and add offspring bacteria cells to
           a list of bacteria in this patient. New offspring do not reproduce.

        4. Reassign the patient's bacteria list to be the list of surviving
           bacteria and new offspring bacteria

        Returns:
            int: The total bacteria population at the end of the update
        """
        # Step 1. is just creating a new list--list comprehension makes it simple
        surviving_bacteria = [bacteria for bacteria in self.bacteria if not bacteria.is_killed()]
        curr_pop_density = len(surviving_bacteria)/self.max_pop
        # Step 3. is also creating a new list
        # --> note: what about when bacteria.reproduce() raises an exception?
        #offspring_bacteria = [bacteria.reproduce(curr_pop_density) for bacteria in surviving_bacteria]
        offspring_bacteria = []
        for bacteria in surviving_bacteria:
            try:
                offspring_bacteria.append(bacteria.reproduce(curr_pop_density))
            except NoChildException:
                continue
        # Step 4. concatenate surviving bacteria and offspring and reassign new list to current bacteria population
        self.bacteria = surviving_bacteria + offspring_bacteria
        return len(self.bacteria)

##########################
# PROBLEM 2
##########################

def calc_pop_avg(populations, n):
    """
    Finds the average bacteria population size across trials at time step n

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria in trial i at time step j

    Returns:
        float: The average bacteria population size at time step n
    """
    # Find number of trials
    num_trials = len(populations)
    # Then the average is the sum of the number of bacteria at every trial i for the same single time step n
    # divided by the number of trials
    return sum([populations[i][n] for i in range(num_trials)])/num_trials


def simulation_without_antibiotic(num_bacteria,
                                  max_pop,
                                  birth_prob,
                                  death_prob,
                                  num_trials):
    """
    Run the simulation and plot the graph for problem 2. No antibiotics
    are used, and bacteria do not have any antibiotic resistance.

    For each of num_trials trials:
        * instantiate a list of SimpleBacteria
        * instantiate a Patient using the list of SimpleBacteria
        * simulate changes to the bacteria population for 300 timesteps,
          recording the bacteria population after each time step. Note
          that the first time step should contain the starting number of
          bacteria in the patient

    Then, plot the average bacteria population size (y-axis) as a function of
    elapsed time steps (x-axis) You might find the make_one_curve_plot
    function useful.

    Args:
        num_bacteria (int): number of SimpleBacteria to create for patient
        max_pop (int): maximum bacteria population for patient
        birth_prob (float in [0, 1]): maximum reproduction
            probability
        death_prob (float in [0, 1]): maximum death probability
        num_trials (int): number of simulation runs to execute

    Returns:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria in trial i at time step j
    """
    # Simulation
    time_steps = 300
    # list comprehension makes list of lists such that index [i][j] is the number of bacteria at trial i and time-step j
    populations_bacteria = [[0 for j in range(time_steps)] for i in range(num_trials)]
    for trial in range(num_trials):
        # instantiate the Bacteria and Patient using provided arguments
        patient_bacteria = [SimpleBacteria(birth_prob, death_prob) for bacteria in range(num_bacteria)]
        patient = Patient(patient_bacteria, max_pop)
        # at time-step dt = 0, the bacteria population in Patient is num_bacteria
        populations_bacteria[trial][0] = num_bacteria
        # update the number of bacteria in the Patient at every time-step dt after time t = 0
        for dt in range(1, time_steps):
            populations_bacteria[trial][dt] += patient.update()

    # Capture the average Bacteria population at current time step to use for plotting
    average_bacteria = [calc_pop_avg(populations_bacteria, dt) for dt in range(time_steps)]

    # Plotting
    make_one_curve_plot(x_coords=list(range(time_steps)), y_coords=average_bacteria,
                        x_label="Time-step", y_label="Average Population", title="Without Antibiotic")
    return populations_bacteria


# When you are ready to run the simulation, uncomment the next line
populations = simulation_without_antibiotic(100, 1000, 0.1, 0.025, 20)

##########################
# PROBLEM 3
##########################

def calc_pop_std(populations, t):
    """
    Finds the standard deviation of populations across different trials
    at time step t by:
        * calculating the average population at time step t
        * compute average squared distance of the data points from the average
          and take its square root

    You may not use third-party functions that calculate standard deviation,
    such as numpy.std. Other built-in or third-party functions that do not
    calculate standard deviation may be used.

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria present in trial i at time step j
        t (int): time step

    Returns:
        float: the standard deviation of populations across different trials at
             a specific time step
    """
    # Get num_trials n
    n = len(populations)
    # mean population at time t
    mean_t = calc_pop_avg(populations, t)
    # average squared distances from mean at time t
    avg_sq_dist = [(populations[i][t] - mean_t)**2 for i in range(n)]
    # standard deviation
    sigma = math.sqrt(sum(avg_sq_dist)/n)
    return sigma


def calc_95_ci(populations, t):
    """
    Finds a 95% confidence interval around the average bacteria population
    at time t by:
        * computing the mean and standard deviation of the sample
        * using the standard deviation of the sample to estimate the
          standard error of the mean (SEM)
        * using the SEM to construct confidence intervals around the
          sample mean

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria present in trial i at time step j
        t (int): time step

    Returns:
        mean (float): the sample mean
        width (float): 1.96 * SEM

        I.e., you should return a tuple containing (mean, width)
    """
    # Compute the standard deviation and the mean of the sample (populations) at time-step t
    sigma = calc_pop_std(populations, t)
    mu = calc_pop_avg(populations, t)
    n = len(populations)
    sem = sigma/math.sqrt(n)
    width = 1.96*sem
    return mu, width


##########################
# PROBLEM 4
##########################

class ResistantBacteria(SimpleBacteria):
    """A bacteria cell that can have antibiotic resistance."""

    def __init__(self, birth_prob, death_prob, resistant, mut_prob):
        """
        Args:
            birth_prob (float in [0, 1]): reproduction probability
            death_prob (float in [0, 1]): death probability
            resistant (bool): whether this bacteria has antibiotic resistance
            mut_prob (float): mutation probability for this
                bacteria cell. This is the maximum probability of the
                offspring acquiring antibiotic resistance
        """
        # A ResistantBacteria is a SimpleBacteria with additional attributes
        SimpleBacteria.__init__(self, birth_prob, death_prob)
        self.resistant = resistant
        self.mut_prob = mut_prob

    def get_resistant(self):
        """Returns whether the bacteria has antibiotic resistance"""
        return self.resistant

    def is_killed(self):
        """Stochastically determines whether this bacteria cell is killed in
        the patient's body at a given time step.

        Checks whether the bacteria has antibiotic resistance. If resistant,
        the bacteria dies with the regular death probability. If not resistant,
        the bacteria dies with the regular death probability / 4.

        Returns:
            bool: True if the bacteria dies with the appropriate probability
                and False otherwise.
        """
        if self.get_resistant():
            return random.random() <= self.death_prob
        else:
            return random.random() <= self.death_prob / 4

    def reproduce(self, pop_density):
        """
        Stochastically determines whether this bacteria cell reproduces at a
        time step. Called by the update() method in the TreatedPatient class.

        A surviving bacteria cell will reproduce with probability:
        self.birth_prob * (1 - pop_density).

        If the bacteria cell reproduces, then reproduce() creates and returns
        an instance of the offspring ResistantBacteria, which will have the
        same birth_prob, death_prob, and mut_prob values as its parent.

        If the bacteria has antibiotic resistance, the offspring will also be
        resistant. If the bacteria does not have antibiotic resistance, its
        offspring have a probability of self.mut_prob * (1-pop_density) of
        developing that resistance trait. That is, bacteria in less densely
        populated environments have a greater chance of mutating to have
        antibiotic resistance.

        Args:
            pop_density (float): the population density

        Returns:
            ResistantBacteria: an instance representing the offspring of
            this bacteria cell (if the bacteria reproduces). The child should
            have the same birth_prob, death_prob values and mut_prob
            as this bacteria. Otherwise, raises a NoChildException if this
            bacteria cell does not reproduce.
        """
        # If the parent reproduces, return the ResistantBacteria instance with appropriate attributes
        if random.random() <= self.birth_prob * (1 - pop_density):
            # assume the parent (and hence the offspring) is resistant
            offspring_resistant = self.resistant
            # if the parent is not resistant, the offspring will be with probability self.mut_prob * (1 - pop_density)
            if not self.resistant:
                offspring_resistant = random.random() <= self.mut_prob * (1 - pop_density)
            return ResistantBacteria(self.birth_prob, self.death_prob, offspring_resistant, self.mut_prob)
        # otherwise, raise the same exception as before
        else:
            raise NoChildException

class TreatedPatient(Patient):
    """
    Representation of a treated patient. The patient is able to take an
    antibiotic and his/her bacteria population can acquire antibiotic
    resistance. The patient cannot go off an antibiotic once on it.
    """
    def __init__(self, bacteria, max_pop):
        """
        Args:
            bacteria: The list representing the bacteria population (a list of
                      bacteria instances)
            max_pop: The maximum bacteria population for this patient (int)

        This function should initialize self.on_antibiotic, which represents
        whether a patient has been given an antibiotic. Initially, the
        patient has not been given an antibiotic.

        Don't forget to call Patient's __init__ method at the start of this
        method.
        """
        Patient.__init__(self, bacteria, max_pop)
        self.on_antibiotic = False

    def set_on_antibiotic(self):
        """
        Administer an antibiotic to this patient. The antibiotic acts on the
        bacteria population for all subsequent time steps.
        """
        self.on_antibiotic = True

    def get_resist_pop(self):
        """
        Get the population size of bacteria cells with antibiotic resistance

        Returns:
            int: the number of bacteria with antibiotic resistance
        """
        return sum([1 for bacteria in self.bacteria if bacteria.get_resistant()])

    def update(self):
        """
        Update the state of the bacteria population in this patient for a
        single time step. update() should execute these actions in order:

        1. Determine whether each bacteria cell dies (according to the
           is_killed method) and create a new list of surviving bacteria cells.

        2. If the patient is on antibiotics, the surviving bacteria cells from
           (1) only survive further if they are resistant. If the patient is
           not on the antibiotic, keep all surviving bacteria cells from (1)

        3. Calculate the current population density. This value is used until
           the next call to update(). Use the same calculation as in Patient

        4. Based on this value of population density, determine whether each
           surviving bacteria cell should reproduce and add offspring bacteria
           cells to the list of bacteria in this patient.

        5. Reassign the patient's bacteria list to be the list of survived
           bacteria and new offspring bacteria

        Returns:
            int: The total bacteria population at the end of the update
        """
        surviving_bacteria = [bacteria for bacteria in self.bacteria if not bacteria.is_killed()]
        # Check whether patient is on antibiotics and whether bacteria is resistant and update surviving bacteria
        if self.on_antibiotic:
            # making sure bacteria in bacteria list is a ResistantBacteria
            # --> note: could also write an abstract method for SimpleBacteria and calls to get_resistant() would simply
            #           use the ResistantBacteria implementation...
            surviving_bacteria = [bacteria for bacteria in surviving_bacteria if isinstance(bacteria, ResistantBacteria)
                                  if bacteria.get_resistant()]
            # --> note: wait no this doesn't work because of the second call to bacteria.is_killed()...it essentially
            #           guarantees that no bacteria survive after you introduce the antibiotic in the simulation...
            #           the list should be the bacteria that were resistant if they survived, NOT the bacteria that
            #           survived, were resistant, and then survived again (essentially none)
            #surviving_bacteria = [bacteria for bacteria in surviving_bacteria if bacteria.get_resistant()
            #                      if not bacteria.is_killed()]
        # calculate current bacteria population density
        curr_pop_density = len(surviving_bacteria)/self.max_pop
        # determine whether surviving bacteria reproduce or not
        offspring_bacteria = []
        for bacteria in surviving_bacteria:
            # try to add bacteria offspring to the offspring list if parents reproduce
            try:
                offspring_bacteria.append(bacteria.reproduce(curr_pop_density))
            # if they don't reproduce, the NoChildException is raised and we continue to check next surviving bacteria
            except NoChildException:
                continue
        # merge the two resulting lists as the new bacteria list and return the number of surviving bacteria
        self.bacteria = surviving_bacteria + offspring_bacteria
        return len(self.bacteria)

##########################
# PROBLEM 5
##########################

def simulation_with_antibiotic(num_bacteria,
                               max_pop,
                               birth_prob,
                               death_prob,
                               resistant,
                               mut_prob,
                               num_trials):
    """
    Runs simulations and plots graphs for problem 4.

    For each of num_trials trials:
        * instantiate a list of ResistantBacteria
        * instantiate a patient
        * run a simulation for 150 timesteps, add the antibiotic, and run the
          simulation for an additional 250 timesteps, recording the total
          bacteria population and the resistance bacteria population after
          each time step

    Plot the average bacteria population size for both the total bacteria
    population and the antibiotic-resistant bacteria population (y-axis) as a
    function of elapsed time steps (x-axis) on the same plot. You might find
    the helper function make_two_curve_plot helpful

    Args:
        num_bacteria (int): number of ResistantBacteria to create for
            the patient
        max_pop (int): maximum bacteria population for patient
        birth_prob (float int [0-1]): reproduction probability
        death_prob (float in [0, 1]): probability of a bacteria cell dying
        resistant (bool): whether the bacteria initially have
            antibiotic resistance
        mut_prob (float in [0, 1]): mutation probability for the
            ResistantBacteria cells
        num_trials (int): number of simulation runs to execute

    Returns: a tuple of two lists of lists, or two 2D arrays
        populations (list of lists or 2D array): the total number of bacteria
            at each time step for each trial; total_population[i][j] is the
            total population for trial i at time step j
        resistant_pop (list of lists or 2D array): the total number of
            resistant bacteria at each time step for each trial;
            resistant_pop[i][j] is the number of resistant bacteria for
            trial i at time step j
    """
    # Simulation
    time_steps = 400
    # 2d arrays such that populations[i][j] is the number of bacteria at the i-th trial and j-th time-step
    populations = [[0 for j in range(time_steps)] for i in range(num_trials)]
    resist_pop = [[0 for j in range(time_steps)] for i in range(num_trials)]
    # for every trial, instantiate a list of ResistantBacteria and a Patient and then update the simulation for the
    # appropriate number of time-steps--here, will introduce antibiotic after the first 150
    for i in range(num_trials):
        patient_bacteria = [ResistantBacteria(birth_prob, death_prob, resistant, mut_prob)
                            for bacteria in range(num_bacteria)]
        patient = TreatedPatient(patient_bacteria, max_pop)
        # at time-step dt = 0, the Patient has num_bacteria bacteria (0 resistant bacteria at the beginning of trials)
        populations[i][0] = num_bacteria
        # for every time-step after dt = 0, update both the total population and the resistant bacteria population
        for dt in range(1, time_steps):
            # if we've passed 150 time-steps, give patient the antibiotic, and record populations as before
            if dt == 150:
                patient.set_on_antibiotic()
            populations[i][dt] += patient.update()
            resist_pop[i][dt] += patient.get_resist_pop()

    # Capture average bacteria population for both total and resistant bacteria populations to use for plotting
    average_total_pop = [calc_pop_avg(populations, dt) for dt in range(time_steps)]
    average_resistant_pop = [calc_pop_avg(resist_pop, dt) for dt in range(time_steps)]

    # Plotting
    make_two_curve_plot(x_coords=list(range(time_steps)), y_coords1=average_total_pop, y_coords2=average_resistant_pop,
                        y_name1="Total", y_name2="Resistant", x_label="Time-step", y_label="Average Population",
                        title="With an Antibiotic")
    return populations, resist_pop


# When you are ready to run the simulations, uncomment the next lines one
# at a time
# Simulation A
total_pop_a, resistant_pop_a = simulation_with_antibiotic(num_bacteria=100,
                                                      max_pop=1000,
                                                      birth_prob=0.3,
                                                      death_prob=0.2,
                                                      resistant=False,
                                                      mut_prob=0.8,
                                                      num_trials=50)

# Calculate the 95% confidence interval for simulation A at dt=299 for the total and resistant bacteria populations
time_step = 299
mu_a, width_a = calc_95_ci(total_pop_a, time_step)
print("Simulation A 95% Confidence Intervals at Time-step = " + str(time_step), end="\n")
print("\tTotal bacteria population: " + str(mu_a) + " +/- " + str(width_a) + ".")
mu_a_res, width_a_res = calc_95_ci(resistant_pop_a, time_step)
print("\tResistant bacteria population: " + str(mu_a_res) + " +/- " + str(width_a_res) + ".\n\n")

# Simulation B
total_pop_b, resistant_pop_b = simulation_with_antibiotic(num_bacteria=100,
                                                      max_pop=1000,
                                                      birth_prob=0.17,
                                                      death_prob=0.2,
                                                      resistant=False,
                                                      mut_prob=0.8,
                                                      num_trials=50)

# Calculate the 95% confidence interval for simulation B at dt=299 for the total and resistant bacteria populations
mu_b, width_b = calc_95_ci(total_pop_b, time_step)
print("Simulation B 95% Confidence Intervals at Time-step = " + str(time_step), end="\n")
print("\tTotal bacteria population: " + str(mu_b) + " +/- " + str(width_b) + ".")
mu_b_res, width_b_res = calc_95_ci(resistant_pop_b, time_step)
print("\tResistant bacteria population: " + str(mu_b_res) + " +/- " + str(width_b_res) + ".\n")
