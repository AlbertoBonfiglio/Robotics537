#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy import stats

from classes.armedbandit2 import ArmedBandit2 #, Arm
from classes.armedbandit3 import ArmedBandit3 #, Arm
from classes.qlearner import Explorer 


def runArmedBandit2Test():
    # run 10 times
    #    
    _mu = [1, 1.5, 2, 2, 1.75]
    _sigma = [5, 1, 1, 2, 10]

    _value = np.mean(_mu)
    runs = 5
    epochs = 500

    bandit = ArmedBandit2(5, mu=_mu, sigma=_sigma, startValue=_value)
    
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
    runs = 10
    epochs = 500
   
    bandit = ArmedBandit3()
   
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    x = np.arange(epochs)
    
    ax = axs[0,0]   
   
    accumulator = []
    errors = []
    y = []
    for i in range(runs):
        start = dt.datetime.now()
        accumulator.append(bandit.performOneArmRobberyEGreedy(epochs=epochs, iterations=20, epsilon=0))
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
        accumulator.append(bandit.performOneArmRobberyEGreedy(epochs=epochs, iterations=20, epsilon=0.1))
        print('Run 10-0.1 #' + str(i) + ' took ' + str((dt.datetime.now() - start).total_seconds()) + 'seconds') 

    accumulator2 = np.array(accumulator)
    for i in range(len(accumulator[0])):
        errors.append(stats.sem(accumulator2[: ,i], axis=None, ddof=0))
        y.append(np.mean(accumulator2[: ,i]))

    ax.plot(x, y, 'g')
    ax.errorbar(x, y, yerr=errors) #, fmt='o')




    fig.suptitle('Variable errorbars')

    plt.show()


def runQLearningTest():
    runs = 1500
    epochs = 500
    _epsilon=0.1
    _alpha=0.5
    _gamma=0.9
    
    
    bandit = Explorer()
   
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
    x = np.arange(epochs)
    
    ax = axs[0]   
    accumulator = []
    errors = []
    y = []
    for i in range(runs):
        start = dt.datetime.now()
        accumulator.append(bandit.findPath(epochs, 0, _alpha, _gamma))
        print('Run 10-0 #' + str(i) + ' took ' + str((dt.datetime.now() - start).total_seconds()) + ' seconds') 

    accumulator2 = np.array(accumulator)
    for i in range(len(accumulator[0])):
        errors.append(stats.sem(accumulator2[: ,i], axis=None, ddof=0))
        y.append(np.mean(accumulator2[: ,i]))
    ax.plot(x, y, 'b', label='Greedy', linewidth=2)
    ax.errorbar(x, y, yerr=errors, ecolor='g') #, fmt='o')
    ax.legend(loc='upper left', shadow=True)
    ax.set_ylabel('Reward')
    ax.set_xlabel('Steps {0}, alpha {1}, gamma {2}'.format(epochs, _alpha, _gamma))
    #--------------------------------------------
    
    ax = axs[1]   
    accumulator = []
    errors = []
    y = []
    for i in range(runs):
        start = dt.datetime.now()
        accumulator.append(bandit.findPath(epochs, _epsilon, _alpha, _gamma))
        print('Run 10-0 #' + str(i) + ' took ' + str((dt.datetime.now() - start).total_seconds()) + ' seconds') 

    accumulator2 = np.array(accumulator)
    for i in range(len(accumulator[0])):
        errors.append(stats.sem(accumulator2[: ,i]/2, axis=None, ddof=0))
        y.append(np.mean(accumulator2[: ,i]))
    
    ax.plot(x, y, 'r--', label='e-Greedy', linewidth=2)
    ax.errorbar(x, y, yerr=errors, ecolor='r') #, fmt='o')
    ax.legend(loc='upper left', shadow=True)
    ax.set_xlabel('Steps {0}, alpha {1}, gamma {2}'.format(epochs, _alpha, _gamma))
    
    _title = 'Q-Learning - Greedy vs e-Greedy (e = {0} - {1} Runs)'.format(_epsilon, runs)
    fig.suptitle(_title)


    plt.show()



if __name__ == '__main__':
    
#    runArmedBandit2Test()

    runArmedBandit3Test()

#    runQLearningTest()