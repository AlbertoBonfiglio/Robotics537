#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy import stats

from classes.armedbandit2 import ArmedBandit2 #, Arm
from classes.armedbandit3 import ArmedBandit3 #, Arm



def runArmedBandit2Test():
    # run 10 times
    #    
    _mu = [1, 1.5, 2, 2, 1.75]
    _sigma = [5, 1, 1, 2, 10]

    _value = np.mean(_mu)
    runs = 5
    epochs = 500

    bandit = ArmedBandit2(5, mu=_mu, sigma=_sigma, startValue=_value)
    accumulator = []
    
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    x = np.arange(epochs)
    
    ax = axs[0,0]   
   
    accumulator = []
    errors = []
    y = []
    for i in range(runs):
        start = dt.datetime.now()
        accumulator.append(bandit.performOneArmRobberyEGreedy(epochs=epochs, iterations=10, epsilon=0))
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
        accumulator.append(bandit.performOneArmRobberyEGreedy(epochs=epochs, iterations=10, epsilon=0.1))
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
        accumulator.append(bandit.performOneArmRobberyEGreedy(epochs=epochs, iterations=100, epsilon=0))
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
        accumulator.append(bandit.performOneArmRobberyEGreedy(epochs=epochs, iterations=100, epsilon=0.1))
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


def runArmedBandit3Test():
    pass

if __name__ == '__main__':
    
    runArmedBandit2Test()

    runArmedBandit3Test()
