# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: Gabriel Munoz
# Collaborators (discussion): None
# Time: Tuesday, September 14, 2021 - Thursday, September 23, 2021 (Thursday, September 29, 2021 for the write-up)

import pylab
import re
# use these instead of pylab in the future
#import matplotlib.pyplot as plt
#import numpy as np

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = pylab.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

"""
End helper code
"""

def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    # Make sure x and y are of the same length before generating models
    assert len(x) == len(y), "x and y arrays are not of the same length N"
    models = [pylab.polyfit(x, y, deg) for deg in degs]
    return models

def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    return 1 - (sum((y-estimated)**2)/sum((y-y.mean())**2))

def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    # Start by creating a new figure and plotting the x and y data, then plot and evaluate each model one by one
    for model in models:
        pylab.figure()
        pylab.plot(x, y, 'bo', label="Data")
        # first we need to generate predicted y values using the models and x data
        y_preds = pylab.polyval(model, x)
        # calculate r-squared for current model
        r_sqrd = r_squared(y, y_preds)
        # find the model's degree (number of coefficients - 1)
        degree = len(model) - 1
        # now we can plot the data and model--if model is linear (degree = 1), also plot the SE/slope
        pylab.plot(x, y_preds, 'r-', label="Model")
        pylab.xlabel("Time in Years")
        pylab.ylabel("Degrees in Celsius (C)")
        if degree == 1:
            pylab.title("Fit of degree " + str(degree) + ", R2 = " + str(round(r_sqrd, 5))
                        + ", SE/slope = " + str(round(se_over_slope(x, y, y_preds, model), 5)))
        else:
            pylab.title("Fit of degree " + str(degree) + ", R2 = " + str(round(r_sqrd, 5)))
        pylab.legend(loc="best")

def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    # Make a 49 x 21 array where every column is each city's 49 average yearly temperatures, then average rows across
    # columns--i.e. for all 49 years, add all 21 temperatures for each year and divide the sum by 21 to get 49 averages
    # --> national yearly temperatures
    all_cities = pylab.array([[climate.get_yearly_temp(city, year).mean() for city in multi_cities] for year in years])
    nat_yearly_temps = all_cities.mean(axis=1)
    return nat_yearly_temps
    # more concise, but MUCH less readable
    # nat_yearly_temps = pylab.array([pylab.array([climate.get_yearly_temp(city, year).mean()
    #                                             for city in multi_cities]).mean() for year in years])

def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    return pylab.array([y[i - window_length + 1:i + 1].mean() if i - window_length + 1 >= 0
                        else y[0:i + 1].mean() for i in range(len(y))])

def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    return pylab.sqrt(((y-estimated)**2).mean())

def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """
    # Initialize empty list for standard deviations
    std_devs_years = []
    # for every year (leap or not)
    for year in years:
        # need a 2d array of 21 cities x 365/366 days of the year, average across columns to get 1d array
        # and then take the standard deviation of that year's daily temperatures (city average) and append to list
        daily_temps_year = pylab.array([climate.get_yearly_temp(city, year) for city in multi_cities])
        daily_temps_year_avg = daily_temps_year.mean(axis=0)
        std_devs_years.append(pylab.std(daily_temps_year_avg))

    # asarray() converts the list, array() makes an array and copies the data from the list to it
    # (similar to shallow vs deep copy, respectively)
    return pylab.asarray(std_devs_years)

def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    # Start by creating a new figure and plotting the x and y data, and then plot and evaluate each model one by one
    for model in models:
        pylab.figure()
        pylab.plot(x, y, 'bo', label="Data")
        # first we need to generate predicted y values using the models and x data
        y_preds = pylab.polyval(model, x)
        # calculate root mean squared error for current model
        rmserror = rmse(y, y_preds)
        # find the model's degree (number of coefficients - 1)
        degree = len(model) - 1
        # now we can plot the data and model
        pylab.plot(x, y_preds, 'r-', label="Model")
        pylab.xlabel("Time in Years")
        pylab.ylabel("Degrees in Celsius (C)")
        pylab.title("Fit of degree " + str(degree) + ", RMSE = " + str(round(rmserror, 5)))
        pylab.legend(loc="best")

if __name__ == '__main__':

    # Part A.4
    # Load in the raw climate data
    climate_data = Climate("data.csv")
    # # A.4.I
    # # Generate samples of data points (x, y)
    # # x will be the years 1961-2009, the training interval
    # x = pylab.array(TRAINING_INTERVAL)
    # # y will be the daily temperature in New York on January 10th of every year from 1961 to 2009
    # y = pylab.array([climate_data.get_daily_temp('NEW YORK', 1, 10, year) for year in x])
    # # fit a polynomial of degree 1 to the data, evaluate the model, and plot the regression results
    # models = generate_models(x, y, [1])
    # evaluate_models_on_training(x, y, models)
    #
    # # A.4.II
    # # Generate samples x and y--x will be the same, but y this time will be the average YEARLY temperature in New York
    # x_2 = pylab.array(TRAINING_INTERVAL)
    # # Use Climate method to get all temperatures for a year and city and take the mean of each year's array
    # y_2 = pylab.array([climate_data.get_yearly_temp('NEW YORK', year).mean() for year in x_2])
    # # fit a polynomial of degree 1 to the data, evaluate the model, and plot the regression results
    # models_2 = generate_models(x_2, y_2, [1])
    # evaluate_models_on_training(x_2, y_2, models_2)
    #
    # # Part B
    # # Incorporating more data--will now average the yearly average temperatures of all 21 cities
    # x_3 = pylab.array(TRAINING_INTERVAL)
    # # National yearly temperatures
    # y_3 = gen_cities_avg(climate_data, CITIES, TRAINING_INTERVAL)
    # # fit a polynomial of degree 1 to the data, evaluate the model, and plot the regression results
    # models_3 = generate_models(x_3, y_3, [1])
    # evaluate_models_on_training(x_3, y_3, models_3)
    #
    # # Part C
    # # Incorporating more data--will now average the yearly average temperatures of all 21 cities
    # x_4 = pylab.array(TRAINING_INTERVAL)
    # # National yearly temperatures transformed to generate moving averages
    # y_4 = moving_average(gen_cities_avg(climate_data, CITIES, TRAINING_INTERVAL), window_length=5)
    # # fit a polynomial of degree 1 to the data, evaluate the model, and plot the regression results
    # models_4 = generate_models(x_4, y_4, [1])
    # evaluate_models_on_training(x_4, y_4, models_4)
    #
    # # Part D.2
    # # Predicting the Future
    #
    # # 2.I Generate more models
    # # we'll be using 5-year moving averages from 1961-2009 for training data
    # x_train = pylab.array(TRAINING_INTERVAL)
    # y_train = moving_average(gen_cities_avg(climate_data, CITIES, TRAINING_INTERVAL), window_length=5)
    # # will fit polynomials of degree 1, 2, and 20 to the training data and then evaluate and plot
    # models_train = generate_models(x_train, y_train, [1, 2, 20])
    # evaluate_models_on_training(x_train, y_train, models_train)
    #
    # # 2.II Predict the result
    # # use the 5-year moving averages from 2010-2015 as test data
    # x_test = pylab.array(TESTING_INTERVAL)
    # y_test = moving_average(gen_cities_avg(climate_data, CITIES, TESTING_INTERVAL), window_length=5)
    # # now use the models fit on the training data to predict the values for weather in the test interval
    # # i.e. evaluate the models generated from the training data on the test data and plot them
    # evaluate_models_on_testing(x_test, y_test, models_train)
    #
    # # # what if we had generated models using the A.4.II data (average annual temperature of NYC in 1961-2009)?
    # # # how would prediction results for 2010-2015 have changed?
    # models_train_2 = generate_models(x_2, y_2, [1, 2, 20])
    # evaluate_models_on_testing(x_test, y_test, models_train_2)

    # Part E
    # Modeling Extreme Temperatures
    x_ext_temps = pylab.array(TRAINING_INTERVAL)
    # standard deviations across cities for every year in the training interval
    std_devs = gen_std_devs(climate_data, CITIES, TRAINING_INTERVAL)
    # compute moving averages
    y_ext_temps = moving_average(std_devs, window_length=5)
    # fit degree-1 poly to data, perform linear regression, and plot the results
    models_ext_temp = generate_models(x_ext_temps, y_ext_temps, [1])
    evaluate_models_on_training(x_ext_temps, y_ext_temps, models_ext_temp)
