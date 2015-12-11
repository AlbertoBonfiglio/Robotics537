#!/usr/bin/env python

import numpy as np
from scipy import stats
import random
import math

import matplotlib.pyplot as plt

#matplotlib inline

class Arm(object):
    pullCount = 0
    reward_payout = 0.0
    reward_variance = 0.0
    current_value = 0.0

    probability = 0.0

    def __init__(self, mu=1, sigma=5, av=0.0):
        pullCount = 0
        self.mu = mu        #mean
        self.sigma = sigma  # variance
        self.current_value = 110#av
        self.probability = np.random.normal(self.mu, self.sigma)

    def pullArm(self):
        self.pullCount += 1
        n = self.pullCount

        value = self.current_value

        self.current_value = ((n -1)/float(n)) * value + (1 / float(n)) * self.getReward()

        return self.current_value


    def getReward(self): #fix this
        #reward = 0
        #for i in range(10):
        #    r = np.random.normal(self.mu, self.sigma)
        #    if r < self.probability:
        #        reward += 1

        #return reward
        return np.random.normal(self.mu, math.sqrt(self.sigma))





class ArmedBandit2(object):

    n = 10
    _mu = [1, 1.5, 2, 2, 1.75]
    _sigma = [5, 1, 1, 2, 10]

    arms = []

    #initialize memory array; has 1 row defaulted to random action index
    av = []

    def __init__(self, n=5, mu = [1, 1.5, 2, 2, 1.75], sigma=[5, 1, 1, 2, 10]):
        try:
            self.n = n

            self._mu = mu
            self._sigma = sigma

            for count in range(len(self._mu)):
                obj = Arm(self._mu[count], self._sigma[count], np.mean(self._mu))
                self.arms.append(obj)

            #self.av = np.array([n-1, np.mean(self._mu)]).reshape(1, 2)
            self.av =  np.array([0, 100])
            self.av =  np.concatenate((self.av, [[1, 100]]), axis=0)
            self.av =  np.concatenate((self.av, [[2, 100]]), axis=0)
            self.av =  np.concatenate((self.av, [[3, 100]]), axis=0)
            self.av =  np.concatenate((self.av, [[4, 100]]), axis=0)
            pass
        except Exception as ex:
            print(ex.args)

    def getArm(self, index):
        return self.arms[index]


    #greedy method to select best arm based on memory array (historical results)
    def bestArm(self, a):
        try:
            bestArm = 0 #just default to 0
            bestMean = 0
            for u in a:
                avg = np.mean(a[np.where(a[:,0] == u[0])][:, 1]) #calc mean reward for each action
                if bestMean < avg:
                    bestMean = avg
                    bestArm = int(u[0])
            print('Best Mean :' + str(bestMean))
            return bestArm
        except Exception as ex:
            print(ex.args)


    def performOneArmRobberyEGreedy(self, epochs=500, epsilon=0.1):
        plt.xlabel("Plays")
        plt.ylabel("Avg Reward")

        self.epsilon = epsilon

        print('starting e-Greedy run ')

        try:
            for i in range(epochs):
                if random.random() > self.epsilon: #greedy arm selection
                    choice = self.bestArm(self.av)

                else: #random arm selection
                    choice = random.randint(0, len(self.arms)-1)

                print('Selected Arm = ' + str(choice))
                thisAV = self.arms[choice].pullArm()
                self.av = np.concatenate((self.av, [[choice, thisAV]]), axis=0)

                #calculate the percentage the correct arm is chosen (you can plot this instead of reward)
                #percCorrect = 100*(len(self.av[np.where(self.av[:,0] == np.argmax(self.arms))])/len(self.av))

                #calculate the mean reward
                runningMean = np.mean(self.av[:,1])

                plt.scatter(i, runningMean)
        except Exception as ex:
            print(ex.args)

        plt.show()


