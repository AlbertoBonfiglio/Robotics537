#!/usr/bin/env python

import numpy as np
from scipy import stats
import random
import math

class Arm(object):
    pullCount = 0
    reward_payout = 0.0
    reward_variance = 0.0
    current_value = 0.0

    probability = 0.0
    iterations = 10

    def __init__(self, mu=1, sigma=5, av=0.0):
        pullCount = 0
        self.mu = mu        #mean
        self.sigma = sigma  # variance
        self.current_value = 3
        self.probability = np.random.normal(self.mu, self.sigma)
        
            
    def pullArm(self, iterations=10):
        self.pullCount += 1
        n = self.pullCount
 
        self.iterations = iterations
 
        value = self.current_value
        self.current_value = ((n -1)/float(n)) * value + (1 / float(n)) * self.getReward()
        
        return self.current_value


    def getReward(self): #fix this
        reward = 0
        #for i in range(self.iterations):
        #    r = +np.random.normal(self.mu, self.sigma)
        #    if r < self.probability:
        #        reward += 1

        for i in range(self.iterations):
            reward = reward + np.random.normal(self.mu, self.sigma)
            

        return reward/self.iterations
        #return np.random.normal(self.mu, math.sqrt(self.sigma))





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
            
            return bestArm
        except Exception as ex:
            print(ex.args)


    def performOneArmRobberyEGreedy(self, epochs=500, iterations=10, epsilon=0.1):
        retval = []

        self.epsilon = epsilon
        try:           #self.av = np.array([n-1, np.mean(self._mu)])
            self.av =  np.array([0, np.mean(self._mu)]).reshape(1, 2)
            self.av =  np.concatenate((self.av, [[1, np.mean(self._mu)]]), axis=0)
            self.av =  np.concatenate((self.av, [[2, np.mean(self._mu)]]), axis=0)
            self.av =  np.concatenate((self.av, [[3, np.mean(self._mu)]]), axis=0)
            self.av =  np.concatenate((self.av, [[4, np.mean(self._mu)]]), axis=0)

            for i in range(epochs):
                if random.random() > self.epsilon: #greedy arm selection
                    choice = self.bestArm(self.av)

                else: #random arm selection
                    choice = random.randint(0, len(self.arms)-1)

                thisAV = self.arms[choice].pullArm(iterations)
                self.av = np.concatenate((self.av, [[choice, thisAV]]), axis=0)

                #calculate the percentage the correct arm is chosen (you can plot this instead of reward)
                #percCorrect = 100*(len(self.av[np.where(self.av[:,0] == np.argmax(self.arms))])/len(self.av))

                #calculate the mean reward
                runningMean = np.mean(self.av[:,1])

                retval.append(runningMean)

            return retval
                
        except Exception as ex:
            print(ex.args)

        plt.show()


