#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy import stats
from classes.armedbandit2 import ArmedBandit2, Arm


if __name__ == '__main__':

    runs = 10

    bandit = ArmedBandit2(5)
    accumulator = []
    
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    x = np.arange(250)
    
    ax = axs[0,0]   
   
    accumulator = []
    errors = []
    y = []
    for i in range(runs):
        start = dt.datetime.now()
        accumulator.append(bandit.performOneArmRobberyEGreedy(epochs=250, iterations=10, epsilon=0))
        print('Run 10-0 #' + str(i) + ' took ' + str((dt.datetime.now() - start).total_seconds()) + ' seconds') 

    accumulator2 = np.array(accumulator)
    for i in range(len(accumulator[0])):
        errors.append(stats.sem(accumulator2[: ,i], axis=None, ddof=0))
        y.append(np.mean(accumulator2[: ,i]))
 
    ax.plot(x, y, 'r')
    ax.errorbar(x, y, yerr=errors) #, fmt='o')
    
    #--------------------------------------------
    accumulator = []
    errors = []
    y = []
    for i in range(runs):
        start = dt.datetime.now()
        accumulator.append(bandit.performOneArmRobberyEGreedy(epochs=250, iterations=10, epsilon=0.1))
        print('Run 10-0.1 #' + str(i) + ' took ' + str((dt.datetime.now() - start).total_seconds()) + 'seconds') 

    accumulator2 = np.array(accumulator)
    for i in range(len(accumulator[0])):
        errors.append(stats.sem(accumulator2[: ,i], axis=None, ddof=0))
        y.append(np.mean(accumulator2[: ,i]))

    ax.plot(x, y, 'g')
    ax.errorbar(x, y, yerr=errors) #, fmt='o')
    
    #--------------------------------------------
    ax = axs[0,1]   

    accumulator = []
    errors = []
    y = []

    for i in range(runs):
        start = dt.datetime.now()
        accumulator.append(bandit.performOneArmRobberyEGreedy(epochs=250, iterations=100, epsilon=0))
        print('Run 100-0 #' + str(i) + ' took ' + str((dt.datetime.now() - start).total_seconds()) + 'seconds') 

    accumulator2 = np.array(accumulator)
    for i in range(len(accumulator[0])):
        errors.append(stats.sem(accumulator2[: ,i], axis=None, ddof=0))
        y.append(np.mean(accumulator2[: ,i]))
 
    ax.plot(x, y, 'r')
    ax.errorbar(x, y, yerr=errors) #, fmt='o')
    
 
    accumulator = []
    errors = []
    y = []
    for i in range(runs):
        start = dt.datetime.now()
        accumulator.append(bandit.performOneArmRobberyEGreedy(epochs=250, iterations=100, epsilon=0.1))
        print('Run 100-0.1 #' + str(i) + ' took ' + str((dt.datetime.now() - start).total_seconds()) + 'seconds') 

    accumulator2 = np.array(accumulator)
    for i in range(len(accumulator[0])):
        errors.append(stats.sem(accumulator2[: ,i], axis=None, ddof=0))
        y.append(np.mean(accumulator2[: ,i]))
 
    ax.plot(x, y, 'g')
    ax.errorbar(x, y, yerr=errors) #, fmt='o')

    
    fig.suptitle('Variable errorbars')

    plt.show()

    pass
