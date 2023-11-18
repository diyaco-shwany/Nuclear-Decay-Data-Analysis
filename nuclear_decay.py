# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 13:05:25

The following program takes in experimentally measured gamma radiation from the
decay of Rubidium-79. It then validates the data, combines the files, fits the
curve and plots the data with the fitted line.
 It then prints the following:
    decay constants of Rubidium-79 and Strontium-79 with their uncertainties.
    The half lives of Rubidium-79 and Strontium-79 with their uncertainties.
    The activity at 90 minutes with its uncertainty in TBq.

It then prints a menu of the following options:
    1) Re-Calculate the results using fmin and the contour plot of chi
    squared values.
    2) Show the contour plot of chi squared values.
    3) Show the activity graph with the non-extreme outliers.
    4) Activity at another time that you can input (in minutes).
    Where the uncertainty is calculated using standard error propogation.
    5) Activity at another time that you can input (in minutes).
    Where the uncertainty is calculated using a monte carlo simulation.
    6) None.
The user can pick any options and the program will print the resulting tasks.

author: q33576
"""
import math as m
import sys
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.optimize as sc
from scipy.constants import Avogadro
import scipy.stats as stat
from sympy import diff, symbols



FILE_NAME_1 = 'Nuclear_data_1.csv'
FILE_NAME_2 = 'Nuclear_data_2.csv'
PLOT_TITLE = 'Activity vs Time plot'
X_LABEL = 'time/s'
Y_LABEL = 'activity/Bq'
AUTO_X_LIMITS = True
X_LIMITS = [0., 10.]
AUTO_Y_LIMITS = True
Y_LIMITS = [0., 10.]
LINE_COLOUR = 'black'
LINE_STYLE = '-'
MARKER_STYLE = 'x'
MARKER_COLOUR = 'red'
GRID_LINES = True
SAVE_FIGURE = True
TRANSPARENT = True
FIGURE_NAME = 'fit_result_Rubidium_decay.png'
FIGURE_RESOLUTION = 400
AUTO_DECAY_GUESSES = False
DECAY_GUESSES = [0.0005, 0.005]



def read_data(file_name):
    """
    Reads in a data file and turns it into a numpy array. Checks if the file
    has exactly 3 columns and returns an error if not. Removes rows of data if
    they contain: nans, infs, negative numbers, zeros in the 3rd column and
    strings. It also converts the 2nd and 3rd column into units of Becquerels
    from Terra Becquerels. It also converts the 1st column into units of
    seconds from minutes

    Parameters
    ----------
    file_name :
        A file of experimentally measured activity of a radioactive source
        measured over some time.

    Returns
    -------
    data_filtered : 2-D numpy array
        The original data set that was inputted but which has now been
        converted into a numpy array and has been validated

    """
    try:

        data = np.genfromtxt(file_name, delimiter = ',', skip_header = 0)
        try:
            data[0,0]
        except IndexError:
            print(f"error! The data file, {file_name}, is empty")
            sys.exit()
            return None
        try:
            data[:,2]
        except IndexError:
            print(f"error! the data file, {file_name},",
                  "doesn't have the correct number of columns: 3")
            sys.exit()
            return None
        try:
            data[:,4]
            print(f"error! the data file, {file_name}, has too many columns. ")
            sys.exit()
        except IndexError:
            counter = 0
            for row in data:
                test = row
                for element in test:
                    try:
                        float(element)
                    except ValueError:

                        data = np.delete(data, counter, axis=0)
                        counter -= 1
                        break
                counter += 1
            data_numeric = data
            invalid_indices = np.array([])
            nan_index = np.where(np.isnan(data_numeric).any(axis = 1))

            inf_index = np.where(np.isinf(data_numeric).any(axis = 1))

            zero_uncertainty_index = np.where(data_numeric[:,2] == 0)

            negative_index = np.where((data_numeric < 0).any(axis = 1))

            invalid_indices = np.unique(np.append(invalid_indices, nan_index))
            invalid_indices = np.unique(np.append(invalid_indices, inf_index))
            invalid_indices = np.unique(np.append(invalid_indices,
                                                  zero_uncertainty_index))
            invalid_indices = np.unique(np.append(invalid_indices, negative_index))


            data_filtered = np.delete(data_numeric, invalid_indices.astype(int), 0)

            data_filtered[:, 0] = data_filtered[:, 0] * 60**2
            data_filtered[:, 1] = data_filtered[:, 1] * 10**12
            data_filtered[:, 2] = data_filtered[:, 2] * 10**12
            return data_filtered

    except FileNotFoundError:

        print(f"file: {file_name} does not exist.")
        return None


def decay_model(time, decay_constant_rubidium, decay_constant_strontium):
    """
    The equation of activity of Rubidium-79 decay:
        number_atoms_rubidium = (initial_atoms_rubidium*
                                 (decay_constant_strontium *
                                  decay_constant_rubidium/
                                        (decay_constant_rubidium -
                                        decay_constant_strontium
                                        ))*(exp(-decay_constant_strontium*time)
                                        - exp(-decay_constant_rubidium*time)))

    Parameters
    ----------
    time : float
        some time value for 0 to infinity
    decay_constant_rubidium : float
        The decay constant of Rubidium-79
    decay_constant_strontium : float
        The decay constant of Strontium-79

    Returns
    -------
    float/array/list
        The activity of Rubidium-79 at the inputted time(s)

    """

    initial_rubidium_moles = 10**-6
    initial_atoms_rubidium = Avogadro * initial_rubidium_moles
    number_atoms_rubidium = (initial_atoms_rubidium*(decay_constant_strontium/
                                    (decay_constant_rubidium -
                                    decay_constant_strontium
                                    ))*(m.e**(-decay_constant_strontium*time)
                                    - m.e**(-decay_constant_rubidium*time)))
    return decay_constant_rubidium*number_atoms_rubidium

def outlier_filter_crude(data):
    """
    Finds the z-scores of every data point in the 2nd column of the inputted
    array and removes any input with z-scores > 3.

    Parameters
    ----------
    data : 2-D numpy array
        A set of experimenttally measured data which may include outliers in
        its 2nd column

    Returns
    -------
    data_without_extremes : 2-D numpy array
        A 2-D numpy array with any data points with z-scores > 3 removed.

    """

    z_scores = stat.zscore(data[:,1])
    data_without_extremes = np.delete(data, np.where(z_scores > 3), axis=0)

    return data_without_extremes

def automatic_guess(data):
    """
    Attempts to estimate the decay constant of Rubidium-79 and Strontium-79
    by estimating their half lives from the inputted data only. It then uses
    the the equation t_(1/2) = ln(2)/(decay_constant) to find the estimate for
    the decay constant

    Parameters
    ----------
    data : 2-D numpy array
        A set of experimentally measure data from the two step decay of
        Strontium-79

    Returns
    -------
    list
        A list of the estimated decay constants of Rubidium-79 and Strontium-79
        in that order (units of s^(-1)).

    """

    activity_peak = np.max(data[:,1])
    counter_1 = 0
    counter_2 = 0
    data = data[data[:,0].argsort()]
    size = len(data[:,1])

    for counter_a in range(size):
        if (data[counter_a, 1] - activity_peak/2) > 0:
            counter_1 = counter_a
            break

    for counter_b in range(size-1, 0, -1):
        if (data[counter_b, 1] - activity_peak/2) > 0:
            counter_2 = counter_b
            break

    strontium_decay_estimate = np.log(2)/data[counter_1,0]
    rubidium_decay_estimate = np.log(2)/(data[counter_2,0] -
                                         2*data[counter_1,0])
    return [rubidium_decay_estimate, strontium_decay_estimate]

def chi_square(parameters, data):
    """
    Returns the chi squared of experimentally measured activity data set
    compared to the predicated values by the activity function.

    Parameters
    ----------
    parameters : list/1-D array
        A list of the decay constants of Rubidium-79 adn Strontium-79,
        respectively
    final_data : A 2-D array of the experimnetally measured activity data set.
        A 3 column array:
            1st column: time/s
            2nd column: Activity/Bq
            3rd column: Activity uncertainty/Bq


    Returns
    -------
    float
    The chi-squared of the inputted data set and decay constants

    """

    decay_1 = parameters[0]
    decay_2 = parameters[1]
    return np.sum((data[:,1] - decay_model(data[:,0], decay_1,
                                                 decay_2))**2/
                                                 data[:,2]**2)


def initial_run_print(minimised_reduced_chi_squared, decay_constants,
                     decay_uncertainties,
                     half_lives, half_life_uncertainties, activity,
                     activity_uncertainty):
    """
    A set of print statements which outputs all the basic calculations
    required.

    Parameters
    ----------
    minimised_reduced_chi_squared : float
        The minimised reduced chi squared of the activity data
    decay_constants : list/1-D array of floats
        Decay constants of Rubidium-79 and Strontium-79 in a list/array
        respectively

    decay_uncertainties : lsit/1-D array of floats
        Decay constant uncertainties of Rubidium-79 and Strontium-79 in a list/
        1-D array respectively
    half_lives : list/1-D array of floats
        Half lives of Rubidium-79 and Strontium-79 in a list/1-D array
    half_life_uncertainties : list/1-D array of floats
        Half life uncertainties of Rubidium-79 and Strontium-79 in a list/1-D
        array respectively
    activity : float
        Activity of Rubidium-79 calculated at some time
    activity_uncertainty : float
        Uncertainty of the calculated activity

    Returns
    -------
    None
    A number of print statements which show the results of the following:7
    - The minimised reduced chi squared of the data after complete validation
    - The decay constant of Rubidium-79 and its uncertainty
    - The decay constant of Strontium-79 and its uncertainty
    - The half life of Rubidium-79 and its uncertaitny
    - The half life of Strontium-79 and its uncertainty
    - The activity of Rubidium-79 and its uncertainty

    """


    print("The results below and the graph were found using"
          " scipy.optimize.curve_fit.")
    print("the minimised reduced chi-squared is: ",
          f"{minimised_reduced_chi_squared:.2f}\n")
    print("the decay constant for Rb-79 is:",
          f"({decay_constants[0]:.3g}",
          f"+/- {decay_uncertainties[0]:.2g}) s^-1\n")
    print("the decay constant for Sr-79 is:",
          f"({decay_constants[1]:.3g}",
          f"+/- {decay_uncertainties[1]:.2g}) s^-1\n")
    print("the half life for Rb-79 is:",
          f"({half_lives[0]:.3g} +/-",
          f"{half_life_uncertainties[0]:.1g}) min(s)\n")
    print("the half life for Sr-79 is:",
          f"({half_lives[1]:.3g} +/-",
          f"{half_life_uncertainties[1]:.1g}) min(s)\n")
    print("the activity at 90 minutes is: ",
          f"({activity:.3g} +/- {activity_uncertainty[0]:.1g}) TBq")

def menu_print():
    """
    Prints a menu of the following options:
        ----------Menu----------
        1) Re-Calculate the results using fmin and the contour plot of
           chi squared values.
        2) Show the contour plot of chi squared values.
        3) Show the activity graph with the non-extreme outliers.
        4) Activity at another time that you can input (in minutes).
           Where the uncertainty is calculated using standard error
           propogation.
        5) Activity at another time that you can input (in minutes).
           Where the uncertainty is calculated using a monte carlo simulation.
        6) None.
    The user then has to pick a number from 1 to 6

    Returns
    -------
    decision : float
        The number from 1 to 6 chosen by the user. The input has been validated
        so only a number from 1 to 6 can be chosen

    """

    print("----------Menu----------")
    print("1) Re-Calculate the results using fmin and the",
          "contour plot of chi squared values.")
    print("2) Show the contour plot of chi squared values.")
    print("3) Show the activity graph with the non-extreme outliers.")
    print("4) Activity at another time that you can input (in minutes).",
        "Where the uncertainty is calculated using standard error propogation.")
    print("5) Activity at another time that you can input (in minutes).",
         "Where the uncertainty is calculated using a monte carlo simulation.")
    print("6) None.\n")

    while True:
        decision = input(str("Please pick an option from 1 to 6: "))
        if decision in ["1", "2", "3", "4", "5", "6"]:
            return decision

        print("------------------------")
        print("Please enter a valid option (1 to 6): ")


def menu_option_result(data, decay_constants,
                       minimised_chi_squared_curve_fit, cov_matrix, outliers, counter_4):
    """
    This function calls the decision menu and then asks which number the user
    chose, it then calls the respecitve function that carries out the tasks
    asked by the user.

    Parameters
    ----------
    data : 2-D array of the experimentally measured activity
        A 3 column array:
            1st column: time/s
            2nd column: Activity/Bq
            3rd column: Activity uncertainty/Bq
    decay_constants : list/1-D array of floats
        list/array of the decay constants of Rubidium-79 and
        Strontium-79 respectively
    minimised_chi_squared_curve_fit : float
        The minised chi squared of the data and the fit which was found using
        curve_fit
    cov_matrix : 2-D array[square matrix]
        The covariance matrix of the two parameters of the fit:
            The decay constant of Rubidium-79 and Strontioum-79
    outliers : 2-D array of floats
        a 3 column array which contains the removed non-extreme outliers:
            1st column: time/s
            2nd column: Activity/Bq
            3rd column: Activity uncertainty/Bq

    Returns
    -------
    The print statments for the respective options and a boolean statement of
    True/False
        If True is returned, the user can pick another option from the menu.
        If False is returned, the program terminates

    """
    decision = menu_print()
    if decision == "1":


        results = fmin(decay_constants, data)
        decay_constants_fmin = results[0]
        meshes = mesh_grid(data, decay_constants[0],
                             decay_constants[1])
        decay_uncertainties = uncertainty_from_contour(meshes[3],
                                                         meshes[0],
                                                         meshes[1],
                                                         results[1])
        minimised_reduced_chi_squared = results[1]/(len(data[:,1]) - 2)
        half_lives = [half_life_calc(decay_constants_fmin[0]),
                      half_life_calc(decay_constants_fmin[1])]
        half_life_uncertainties = [
            half_life_uncertainty(decay_constants_fmin[0],
                                  half_lives[0],
                                  decay_uncertainties[0])
            , half_life_uncertainty(decay_constants_fmin[1],
                                    half_lives[1],
                                    decay_uncertainties[1])]

        print("------------------------")
        print("the minimised reduced chi-squared is: ",
              f"{minimised_reduced_chi_squared:#.2f}\n")
        print("the decay constant for Rb-79 is:",
              f"({decay_constants_fmin[0]:#.3g}",
              f"+/- {decay_uncertainties[0]:#.2g}) s^-1\n")
        print("the decay constant for Sr-79 is:",
              f"({decay_constants_fmin[1]:#.3g}",
              f"+/- {decay_uncertainties[1]:#.2g}) s^-1\n")
        print("the half life for Rb-79 is:",
              f"({half_lives[0]:#.3g} +/-",
              f"{half_life_uncertainties[0]:#.1g}) min(s)\n")
        print("the half life for Sr-79 is:",
              f"({half_lives[1]:#.3g} +/-",
              f"{half_life_uncertainties[1]:#.1g}) min(s)\n")

        return True

    if decision == "2":

        meshes = mesh_grid(data,
                             decay_constants[0],
                             decay_constants[1])
        mesh_colour_plot(meshes[0], meshes[1], meshes[3],
                         minimised_chi_squared_curve_fit,
                         decay_constants[0],
                         decay_constants[1])


        return True

    elif decision == "3":
        decay_uncertainties = np.sqrt(np.diag(cov_matrix))
        #plot(outliers)
        plot_1(data, decay_constants[0], decay_constants[1],
               minimised_chi_squared_curve_fit, decay_uncertainties[0],
               decay_uncertainties[1], outliers)


        return True
        #sys.exit()
    elif decision == "4":
        while True:
            print("------------------------")
            time_input = input(str("Please enter a time in minutes: "))
            try:
                time_input = float(time_input)
                if time_input >= 0:

                    time_converted = time_input * 60
                    activity = decay_model(time_converted,
                                           decay_constants[0], decay_constants[1])
                    activity_converted = activity * 10**-12
                    activity_uncertainty = float(activity_uncertainty_errorprop(
                        decay_constants[0], decay_constants[1], cov_matrix, time_converted))
                    print(f"the activity at {time_input} minutes is: "
                         "({0:.3g}".format(activity_converted),
                         "+/- {0:.1g}) TBq".format(activity_uncertainty))
                    return True
                print("------------------------")
                print("Please enter a positive time in minutes")


            except ValueError:
                print("------------------------")
                print("Please enter a numeric time in minutes")

    elif decision == "5":
        while True:
            print("------------------------")
            time_input = input(str("Please enter a time in minutes: "))
            try:
                time_input = float(time_input)

                if time_input >= 0:
                    time_converted = time_input * 60
                    activity = decay_model(time_converted,
                                           decay_constants[0],
                                           decay_constants[1])
                    activity_converted = activity * 10**-12
                    activity_uncertainty, activity_linspace = activity_uncertainty_montecarlo(
                                                    decay_constants[0],
                                                    decay_constants[1],
                                                    cov_matrix,
                                                    time_converted)
                    print("------------------------")
                    print(f"the activity at {time_input} minutes is: "
                         f"({activity_converted:.3g}",
                         f"+/- {activity_uncertainty:.1g}) TBq")
                    while True:
                        print("------------------------")
                        print('Would you like to see the monte carlo method',
                              'results visualised on a plot?')
                        plot_decision = input(str("[Y/N]: "))
                        if plot_decision.capitalize() == "Y":
                            montecarlo_plot(activity_linspace, counter_4)


                            break
                        if plot_decision.capitalize() == "N":
                            break

                        print("------------------------")
                        print("Please enter [Y/N]")
                    return True


                print("------------------------")
                print("Please enter a positive time in minutes")
            except ValueError:

                    print("------------------------")
                    print("Please enter a time in minutes")


    elif decision == "6":
        return False

def outlier_filter(data, decay_1, decay_2):
    """
    Removes non-extreme outliers and stores them in a separate 2-D array.

    Parameters
    ----------
    data : A 2-D numpy array of floats
        A 3 column array:
            1st column: time/s
            2nd column: Activity/Bq
            3rd column: Activity uncertainty/Bq
    decay_1 : float
        The decay constant of Rubidium-79
    decay_2 : float
        The deccay constant of Strontium-79

    Returns
    -------
    data : A 2-D numpy array of floats
        Same as the inputted array but with outliers removed
    outliers :A 2-D numpy array of floats
        The removed outliers inside their own array

    """

    counter = 0
    sigma = np.mean(data[:,2])
    outliers = np.zeros((0,3))
    for observation in data:
        if abs(observation[1] - (decay_model(observation[0], decay_1,
                                             decay_2))) > 3*sigma:
            temp_outlier = data[counter,:]
            outliers = np.vstack((outliers, temp_outlier))
            data = np.delete(data, counter, axis=0)
            continue

        counter += 1
    return data, outliers


def half_life_calc(decay_constant):
    """
    Calculates the half life of a source given its decay constant using the
    following equation:
        half-life = ln(2)/(decay-constant)


    Parameters
    ----------
    decay_constant : float
        Decay constant of some source

    Returns
    -------
    float
        The half life in minutes

    """

    half_life = np.log(2)/decay_constant
    return half_life / 60

def half_life_uncertainty(decay_constant, half_life,
                          decay_constant_uncertainty):
    """
    Calculates the uncertainty on some half life given the uncertainty of
    the decay constant.

    Parameters
    ----------
    decay_constant : float
        The decay constant of some source(s^(-1))
    half_life : float
        The half life of the same source (minutes)
    decay_constant_uncertainty : float
        The uncertainty of the decay constant(s^(-1))

    Returns
    -------
    uncertainty : float
        The uncertainty on the half life

    """

    uncertainty = (decay_constant_uncertainty/decay_constant) * half_life

    return uncertainty




def mesh_grid(data, decay_rubidium, decay_strontium):
    """
    Creates a mesh grid for the following values:
        Decay constant of Rubidium-79
        Decay constant of Strontium-79
        Chi squared values for all the possible combinations of the decay
        constants
    Parameters
    ----------
    data : A 2-D array of floats
       A 3 column array:
           1st column: time/s
           2nd column: Activity/Bq
           3rd column: Activity uncertainty/Bq
    decay_rubidium : float
        The decay constant of Rubidium-79(s^(-1))
    decay_strontium : float
        The decay constant of Strontium-79(s^(-1))

    Returns
    -------
    x_mesh_2 : 2-D array of floats
        The mesh of Rubidium-79 decay constant
    y_mesh_2 : 2-D array of floats
        The mesh of Strontium-79 decay constant
    final_mesh : 2-D array of floats
        The mesh of chi-squared values
    chi_mesh_1 : 2-D array of floats
        The mesh of chi-squared values

    """

    data = data[data[:,0].argsort()]
    percentage_change = 0.065
    precision = 200
    x_mesh = np.linspace(decay_rubidium * (1 - percentage_change),
                         decay_rubidium * (1 + percentage_change), precision)
    y_mesh = np.linspace(decay_strontium * (1 - percentage_change), decay_strontium *
                         (1 + percentage_change), precision)
    x_mesh_2, y_mesh_2 = np.meshgrid(x_mesh, y_mesh)
    final_mesh = np.zeros((len(y_mesh), len(x_mesh)), dtype=object)
    chi_mesh_1 = np.zeros((len(y_mesh), len(x_mesh)))
    for counter in range(len(data[:,1])):
        rows = data[counter,:]
        chi_mesh_1 += (rows[1] -
                       decay_model(rows[0], x_mesh_2, y_mesh_2))**2/rows[2]**2

    return (x_mesh_2, y_mesh_2, final_mesh, chi_mesh_1)


#pass meshes = [x_mesh, y_mesh, z_mesh], decays = [decay_rubidum, decay_strontium]
def mesh_colour_plot(x_mesh, y_mesh, z_mesh, minimised_chi_squared,
                     decay_rubidium, decay_strontium):
    """
    Plots the chi_mesh and the two decay meshes on a colour contour plot.

    Parameters
    ----------
    x_mesh : A 2-D array of floats

    y_mesh : A 2-D array of floats

    z_mesh : A 2-D array of floats
        The values of a function with the x and y-mesh inputted into it
    minimised_chi_squared : float
        The minimised chi squared of the data and fit
    decay_rubidium : TYPE
        DESCRIPTION.
    decay_strontium : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    fig = plt.figure()

    axis = fig.add_subplot(111)

    chi_min, chi_max = np.abs(z_mesh).min(), np.abs(z_mesh).max()

    contour = axis.contourf(x_mesh, y_mesh, z_mesh, 50,
                          cmap='plasma', vmin=chi_min, vmax=chi_max)
    fig.colorbar(contour)
    axis.set_title(r'contour plot of $\chi^{2}$ values')
    axis.set_xlabel(r'$\lambda_{Rb^{79}}/s^{-1}$')
    axis.set_ylabel(r'$\lambda_{Sr^{79}}/s^{-1}$')
    axis.scatter(decay_rubidium, decay_strontium)
    axis.contour(x_mesh, y_mesh, z_mesh, [minimised_chi_squared + 1],
                cmap = 'ocean_r')
    axis.contour(x_mesh, y_mesh, z_mesh, [minimised_chi_squared + 2.3],
                cmap = 'ocean_r')
    axis.contour(x_mesh, y_mesh, z_mesh, [minimised_chi_squared + 5.99],
                cmap = 'ocean_r')
    plt.tight_layout()
    plt.savefig('chi_squared_contour.png', dpi=FIGURE_RESOLUTION,
                transparent=TRANSPARENT)
    plt.show()




def uncertainty_from_contour(z_mesh, x_mesh, y_mesh, minimised_chi):
    """
    Takes the input of decay meshes and chi squared meshes. Finds the inputs in
    chi mesh which are approximately equal to minimised chi squared + 1. It
    then finds the edges of error ellipse and approximates the errors on the
    decay constants

    Parameters
    ----------
    z_mesh : A 2-D array of floats
        mesh of chi squared values
    x_mesh : A 2-D array of floats
        mesh of Rubidium-79 decay constant
    y_mesh : A 2-D array of floats
        mesh of Strontium-79 decay constant
    minimised_chi : float
        ThE minimised chi squared of the data and fit

    Returns
    -------
    list of floats
        The errors on the decay constants of the Rubidium-79 and Strontium-79

    """

    locator_mesh = z_mesh - (minimised_chi + 1)

    indices = np.argwhere(locator_mesh < 0)
    rows, cols, *_ = np.split(indices, 2, axis=1)



    cols_unique = np.unique(cols)
    rows_unique = np.unique(rows)
    decay_rubidium = []
    decay_strontium = []
    for i in cols_unique:
        decay_rubidium.append(x_mesh[0,i])
    for j in rows_unique:
        decay_strontium.append(y_mesh[j,0])

    decay_rubidium_max = np.max(decay_rubidium)
    decay_rubidium_min = np.min(decay_rubidium)
    decay_strontium_max = np.max(decay_strontium)
    decay_strontium_min = np.min(decay_strontium)

    rubidium_error = (decay_rubidium_max - decay_rubidium_min)/2
    strontium_error = (decay_strontium_max - decay_strontium_min)/2

    return [rubidium_error, strontium_error]

def activity_uncertainty_montecarlo(rubidium_decay, strontium_decay,
                                    cov_matrix, time):
    """
    Utilizes a monte carlo method to find the uncertainty on activity at some
    time

    Parameters
    ----------
    rubidium_decay : float
        The decay constant of Rubidium-79
    strontium_decay : float
        The decay constant of Strontium-79
    cov_matrix : A 2-D array of floats[square matrix]
        The covariance matrix of the RUbidium and Strontium-79 decay constants
    time : float
        some inputted time

    Returns
    -------
    float
        The error on activity in TBq
    activity : A 1-D array
        A linpace of values from the minimum activity to the largest activty
        found by the monte carlo method

    """

    length = int(10e4)
    mean = [rubidium_decay, strontium_decay]

    activity_normalised = npr.multivariate_normal(mean, cov_matrix, size=length)

    rubidium_decay_rn = activity_normalised[:,0]
    strontium_decay_rn =activity_normalised[:,1]


    activity = decay_model(time, rubidium_decay_rn, strontium_decay_rn)

    return np.std(activity * 10**-12), activity

def montecarlo_plot(activity, counter):
    """
    Plots the normal distribution of activties as a histogram.

    Parameters
    ----------
    activity : A 1-D array floats
        The normal distribution of activity values about its mean

    Returns
    -------
    None.

    """
    fig = plt.figure()

    axis = fig.add_subplot(111)
    activity_converted = activity * 10**-12
    sigma = np.std(activity_converted)
    mean = np.mean(activity_converted)
    axis.set_xlabel("Activity/ TBq")
    axis.set_ylabel("probability density")
    axis.set_title('Monte_carlo_plot')
    activity_linspace = np.linspace(np.min(activity_converted),
                                    np.max(activity_converted), 100)
    axis.hist(activity_converted , bins=100,
              density = True)
    gaussian = (1/(np.sqrt(2 * np.pi) * sigma)) * np.exp(-(1/2)*((
        activity_linspace - mean)/sigma)**2)
    axis.plot(activity_linspace, gaussian, color="red")
    figure_name = str('Monte_carlo_plot_at_{0:.3g}_TBq_{1}.png'.format(mean,
                                                                      counter))
    plt.savefig(figure_name, dpi=FIGURE_RESOLUTION,
                transparent=TRANSPARENT)
    plt.show()


def activity_uncertainty_errorprop(rubidium_decay, strontium_decay, cov_matrix,
                                   time):
    """
    Calculates the error on activity at some time using the following equation:
        sigma_activity = g^(T) * V * g
        Where g is a vector and each input of the vector is the partial
        differential of the function deifferentiated with respect to a
        parameter it depends on

        Where V is the variance-covariance matrix pf the parameters

    Parameters
    ----------
    rubidium_decay : float
        The decay constant of Rubidium-79
    strontium_decay : float
        The decay constant of Strontium-79
    cov_matrix : A 2-D array of floats[square matrix]
        The covariance matrix of the decay constants
    time : float
        some inputted time in (minutes)

    Returns
    -------
    float
        The error on activity in TBq

    """

    parameter_1, parameter_2= symbols("parameter_1 parameter_2 ", real=True)
    activity = decay_model(time, parameter_1, parameter_2)

    decay_differentiated_rubidium = diff(activity, parameter_1).subs([(parameter_1,
                                                                       rubidium_decay),
                                                                      (parameter_2,
                                                                       strontium_decay)])
    decay_differentiated_strontium = diff(activity, parameter_2).subs([(parameter_1,
                                                                        rubidium_decay),
                                                                       (parameter_2,
                                                                        strontium_decay)])

    differential_vector_transpose = [decay_differentiated_rubidium,
                                     decay_differentiated_strontium]
    differential_vector = np.transpose(differential_vector_transpose)
    activity_variance = np.dot(np.dot(differential_vector_transpose,
                                      cov_matrix), differential_vector)

    return ((activity_variance)**0.5) * 10**-12
def plot(data):
    """
    Plots scatter of data points

    Parameters
    ----------
    data : A 2-D array of floats
        A 3 column array:
            1st column: time/s
            2nd column: Activity/Bq
            3rd column: Activity uncertainty/Bq

    Returns
    -------
    None.

    """

    plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], fmt='.')
    plt.show()

def plot_1(data, rubidium_decay, strontium_decay, chi_squared,
           strontium_decay_uncertainty
           , rubidium_decay_uncertainty, outliers):
    """
    Plots a graph of the validated data points and the fitted line.

    Parameters
    ----------
    data : A 2-D array of floats
        A 3 column array:
            1st column: time/s
            2nd column: Activity/Bq
            3rd column: Activity uncertainty/Bq

    rubidium_decay : float
        The decay constant of Rubidium-79
    strontium_decay : float
        The decay constant of Strontium-79
    chi_squared : float
        The minimised chi squared of the data and the fitted line
    sr_decay_uncer : float
        Uncertainty on decay constant of Strontium-79
    rb_decay_uncer : float
        Uncertainty on decay constant of Rubidium-79
    outliers : A 2-D array of floats
        The array of non-extreme outliers:
            A 3 column array:
                1st column: time/s
                2nd column: Activity/Bq
                3rd column: Activity uncertainty/Bq

    Returns
    -------
    None.

    """


    figure = plt.figure(figsize=(8, 6))
    if len(outliers) == 0:
        axes_main_plot = figure.add_subplot(211)
        x_data = data[:,0]
        y_data = data[:,1]
        y_uncertainties  = data[:,2]
        axes_main_plot.errorbar(x_data, y_data, yerr=y_uncertainties,
                                fmt=MARKER_STYLE, color=MARKER_COLOUR)
        axes_main_plot.plot(x_data, decay_model(x_data, rubidium_decay,
                                                strontium_decay),
                            color=LINE_COLOUR)
        axes_main_plot.grid(GRID_LINES)
        axes_main_plot.set_title(PLOT_TITLE, fontsize=14)
        axes_main_plot.set_xlabel(X_LABEL)
        axes_main_plot.set_ylabel(Y_LABEL)
        # Fitting details
        degrees_of_freedom = len(x_data) - 2
        reduced_chi_squared = chi_squared / degrees_of_freedom

        axes_main_plot.annotate((r'$\chi^2$ = {0:#4.2f}'.
                                 format(chi_squared)), (1, 0), (-60, -35),
                                xycoords='axes fraction', va='top',
                                textcoords='offset points', fontsize='10')
        axes_main_plot.annotate(('Degrees of freedom = {0:d}'.
                                 format(degrees_of_freedom)), (1, 0), (-147, -55),
                                xycoords='axes fraction', va='top',
                                textcoords='offset points', fontsize='10')
        axes_main_plot.annotate((r'Reduced $\chi^2$ = {0:#4.2g}'.
                                 format(reduced_chi_squared)), (1, 0), (-104, -70),
                                xycoords='axes fraction', va='top',
                                textcoords='offset points', fontsize='10')
        axes_main_plot.annotate(r'Fit: $A_{Rb^{79}} = N_{Sr^{79}}(0)'
                                r'\frac{\lambda_{Rb^{79}}\lambda_{Sr^{79}}}'
                                r'{\lambda_{Rb^{79}}-\lambda_{Sr^{79}}}'
                                r'[\exp(-\lambda_{Sr^{79}}t)-\exp(-\lambda_{Rb^{79}}t)]$', (0, 0), (0, -35),
                                xycoords='axes fraction', va='top',
                                textcoords='offset points')
        axes_main_plot.annotate(('lambda_Rb-79 = {0:#4.3g}'.format(rubidium_decay)), (0, 0),
                                (0, -55), xycoords='axes fraction', va='top',
                                textcoords='offset points', fontsize='10')
        axes_main_plot.annotate(('± {0:#4.3g}'.format(rubidium_decay_uncertainty)),
                                (0, 0), (135, -55), xycoords='axes fraction',
                                va='top', fontsize='10',
                                textcoords='offset points')
        axes_main_plot.annotate(('lambda_Sr-79 = {0:#4.3g}'.format(strontium_decay)), (0, 0),
                                (0, -70), xycoords='axes fraction', va='top',
                                textcoords='offset points', fontsize='10')
        axes_main_plot.annotate(('± {0:#4.3g}'.format(strontium_decay_uncertainty)),
                                (0, 0), (135, -70), xycoords='axes fraction',
                                textcoords='offset points', va='top',
                                fontsize='10')
        # Residuals plot
        residuals = y_data - decay_model(x_data, rubidium_decay, strontium_decay)
        axes_residuals = figure.add_subplot(414)
        axes_residuals.errorbar(x_data, residuals, yerr=y_uncertainties,
                                fmt=MARKER_STYLE, color=MARKER_COLOUR)
        axes_residuals.plot(x_data, 0 * x_data, color=LINE_COLOUR)
        axes_residuals.grid(True)
        axes_residuals.set_title('Residuals', fontsize=14)

        if not AUTO_X_LIMITS:
            axes_main_plot.set_xlim(X_LIMITS)
            axes_residuals.set_xlim(X_LIMITS)
        if not AUTO_Y_LIMITS:
            axes_main_plot.set_ylim(Y_LIMITS)
            axes_residuals.set_ylim(Y_LIMITS)

        if SAVE_FIGURE:
            plt.savefig(FIGURE_NAME,
                        dpi=FIGURE_RESOLUTION, transparent=TRANSPARENT)
        plt.show()

    else:
        axes_main_plot = figure.add_subplot(211)
        x_data = data[:,0]
        y_data = data[:,1]
        y_uncertainties  = data[:,2]
        try:
            axes_main_plot.errorbar(outliers[:,0], outliers[:,1],
                                    yerr = outliers[:,2], fmt=MARKER_STYLE,
                                    color="blue")
        except IndexError:
            axes_main_plot.errorbar(outliers[0], outliers[1],
                                    yerr = outliers[2], fmt=MARKER_STYLE,
                                    color = "blue")
        axes_main_plot.errorbar(x_data, y_data, yerr=y_uncertainties,
                                fmt=MARKER_STYLE, color=MARKER_COLOUR)
        axes_main_plot.plot(x_data, decay_model(x_data, rubidium_decay,
                                                strontium_decay),
                            color=LINE_COLOUR)
        axes_main_plot.grid(GRID_LINES)
        axes_main_plot.set_title(PLOT_TITLE, fontsize=14)
        axes_main_plot.set_xlabel(X_LABEL)
        axes_main_plot.set_ylabel(Y_LABEL)
        # Fitting details
        degrees_of_freedom = len(x_data) - 2
        reduced_chi_squared = chi_squared / degrees_of_freedom

        axes_main_plot.annotate((r'$\chi^2$ = {0:4.2f}'.
                                 format(chi_squared)), (1, 0), (-60, -35),
                                xycoords='axes fraction', va='top',
                                textcoords='offset points', fontsize='10')
        axes_main_plot.annotate(('Degrees of freedom = {0:d}'.
                                 format(degrees_of_freedom)), (1, 0), (-147, -55),
                                xycoords='axes fraction', va='top',
                                textcoords='offset points', fontsize='10')
        axes_main_plot.annotate((r'Reduced $\chi^2$ = {0:4.2f}'.
                                 format(reduced_chi_squared)), (1, 0), (-104, -70),
                                xycoords='axes fraction', va='top',
                                textcoords='offset points', fontsize='10')
        axes_main_plot.annotate(r'Fit: $A_{Rb^{79}} = N_{Sr^{79}}(0)\frac{\lambda_{Rb^{79}}\lambda_{Sr^{79}}}{\lambda_{Rb^{79}}-\lambda_{Sr^{79}}}[\exp(-\lambda_{Sr^{79}}t)-\exp(-\lambda_{Rb^{79}}t)]$', (0, 0), (0, -35),
                                xycoords='axes fraction', va='top',
                                textcoords='offset points')
        axes_main_plot.annotate(('lambda_Rb-79 = {0:#4.2e}'.format(rubidium_decay)), (0, 0),
                                (0, -55), xycoords='axes fraction', va='top',
                                textcoords='offset points', fontsize='10')
        axes_main_plot.annotate(('± {0:#4.2e}'.format(rubidium_decay_uncertainty)),
                                (0, 0), (135, -70), xycoords='axes fraction',
                                va='top', fontsize='10',
                                textcoords='offset points')
        axes_main_plot.annotate(('lambda_Sr-79= {0:#4.2e}'.format(strontium_decay)), (0, 0),
                                (0, -70), xycoords='axes fraction', va='top',
                                textcoords='offset points', fontsize='10')
        axes_main_plot.annotate(('± {0:#4.1e}'.format(strontium_decay_uncertainty)),
                                (0, 0), (135, -55), xycoords='axes fraction',
                                textcoords='offset points', va='top',
                                fontsize='10')
        # Residuals plot
        residuals = y_data - decay_model(x_data, rubidium_decay, strontium_decay)
        axes_residuals = figure.add_subplot(414)
        axes_residuals.errorbar(x_data, residuals, yerr=y_uncertainties,
                                fmt=MARKER_STYLE, color=MARKER_COLOUR)
        axes_residuals.plot(x_data, 0 * x_data, color=LINE_COLOUR)
        axes_residuals.grid(True)
        axes_residuals.set_title('Residuals', fontsize=14)

        if not AUTO_X_LIMITS:
            axes_main_plot.set_xlim(X_LIMITS)
            axes_residuals.set_xlim(X_LIMITS)
        if not AUTO_Y_LIMITS:
            axes_main_plot.set_ylim(Y_LIMITS)
            axes_residuals.set_ylim(Y_LIMITS)

        if SAVE_FIGURE:
            plt.savefig('fit_result_Rubidium_decay_outliers.png',
                        dpi=FIGURE_RESOLUTION, transparent=TRANSPARENT)
        plt.show()

def fmin(decay_constants, data):
    """
    Utilises scipy.optimize.fmin to minise the chi squared of the fitted line.

    Parameters
    ----------
    decay_constants : list
        List containing decay constants of Rubidium and Strontium-79
    data : A 2-D array of floats
        A 3 column array:
            1st column: time/s
            2nd column: Activity/Bq
            3rd column: Activity uncertainty/Bq

    Returns
    -------
    results : ARRAY
        Ouputs of fmin

    """



    results = sc.fmin(chi_square, (decay_constants[0],
                                           decay_constants[1]),
                      args = (data, ), full_output=True, disp=False)

    return results


def curve_fit(decay_values, final_data):
    """
    Utilizes the scipy.optimize.curve_fit function to minimise the chi squared
    of a line to a data

    Parameters
    ----------
    decay_values : list
        A list of the decay constant of Rubidium-79 and Strontium-79
    final_data : A 2-D array of floats
        A 3 column array:
            1st column: time/s
            2nd column: Activity/Bq
            3rd column: Activity uncertainty/Bq

    Returns
    -------
    results : ARRAY
    results of curve_fit

    """

    results = sc.curve_fit(decay_model, final_data[:,0],
                           final_data[:,1],
                           p0=[decay_values[0], decay_values[1]],
                           sigma=final_data[:,2], absolute_sigma=True)

    return  results

def main():
    """
    This is the main function where all other functions are called

    Returns
    -------
    None.

    """
    data_1 = read_data(FILE_NAME_1)
    data_2 = read_data(FILE_NAME_2)

    if len(data_1) == 0:
        print(f"error! the data file, {FILE_NAME_1}, is empty."
              "Please input a valid data file.")

    elif len(data_2) == 0:
        print(f"error! the data file, {FILE_NAME_2}, is empty."
              "Please input a valid data file.")
    else:

        data_raw = np.vstack((data_1, data_2))

        data_sorted = data_raw[data_raw[:,0].argsort()]

        data_filter_crude = outlier_filter_crude(data_sorted)
        if AUTO_DECAY_GUESSES:
            decay_constants_guess = automatic_guess(data_filter_crude)
        else:
            decay_constants_guess = DECAY_GUESSES


        results_initial = curve_fit(decay_constants_guess, data_filter_crude)

        decay_constants_initial = results_initial[0]

        data_filter_fine, outliers= outlier_filter(data_filter_crude,
                                          decay_constants_initial[0],
                                          decay_constants_initial[1],)
        results_final = curve_fit(decay_constants_initial, data_filter_fine)

        decay_constants_final = results_final[0]
        chi_squared_minimised = chi_square(decay_constants_final,
                                           data_filter_fine)
        minimised_reduced_chi_squared = (chi_squared_minimised
                                         /(len(data_filter_fine[:,1]) - 2))

        rubidium_half_life = half_life_calc(decay_constants_final[0])

        strontium_half_life = half_life_calc(decay_constants_final[1])
        half_lives = [rubidium_half_life, strontium_half_life]

        decay_uncertainties = np.sqrt(np.diag(results_final[1]))

        cov_matrix = results_final[1]

        rubidium_half_life_uncertainty = half_life_uncertainty(decay_constants_final[0],
                                                        half_lives[0],
                                                        decay_uncertainties[0])
        strontium_half_life_uncertainty = half_life_uncertainty(
                                                    decay_constants_final[1],
                                                    half_lives[1],
                                                    decay_uncertainties[1])
        half_life_uncertainties = [rubidium_half_life_uncertainty,
                                   strontium_half_life_uncertainty]

        time_90 = 90*60
        activity_90_minutes = 10**-12 * decay_model(time_90,
                                          decay_constants_final[0],
                                          decay_constants_final[1])
        activity_uncertainty = activity_uncertainty_errorprop(decay_constants_final[0],
                                                              decay_constants_final,
                                                              cov_matrix, time_90)
        activity_uncertainty = activity_uncertainty_montecarlo(
                                        decay_constants_final[0],
                                        decay_constants_final[1],
                                        cov_matrix, time_90)
        initial_run_print(minimised_reduced_chi_squared,
                          decay_constants_final,
                          decay_uncertainties,
                          half_lives,
                          half_life_uncertainties, activity_90_minutes,
                          activity_uncertainty)

        plot_1(data_filter_fine, decay_constants_final[0],
               decay_constants_final[1], chi_squared_minimised,
               decay_uncertainties[1], decay_uncertainties[0], np.array([]))
        break_flag = True
        counter = 0
        while break_flag:
            break_flag = menu_option_result(data_filter_fine, decay_constants_final,
                           chi_squared_minimised, cov_matrix, outliers, counter)
            counter += 1

main()
