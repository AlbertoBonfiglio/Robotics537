 #!/usr/bin/env python

import numpy as np
from scipy import stats
import random

import matplotlib.pyplot as plt

#matplotlib inline


class ArmedBandit(object):

    n = 10
    arms = np.random.rand(n)

    #initialize memory array; has 1 row defaulted to random action index
    av = None

    def __init__(self, n=10):
        self.n = n
        self.arms = np.random.rand(n)

        pass


    def reward(self, prob):
        reward = 0
        for i in range(100):
            if random.random() < prob:
                reward += 1
        return reward


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
            print(Exception.args)

    def performOneArmRobberyEGreedy(self, epochs=500, epsilon=0.1):
        plt.xlabel("Plays")
        plt.ylabel("Avg Reward")

        self.epsilon = epsilon

        print('starting e-Greedy run ')
        self.av = np.array([np.random.randint(0, (self.n+1)), 0]).reshape(1, 2) #av = action-value

        for i in range(epochs):
            if random.random() > self.epsilon: #greedy arm selection
                choice = self.bestArm(self.av)
                thisAV = np.array([[choice, self.reward(self.arms[choice])]])
                self.av = np.concatenate((self.av, thisAV), axis=0)

            else: #random arm selection
                choice = np.where(self.arms == np.random.choice(self.arms))[0][0]
                thisAV = np.array([[choice, self.reward(self.arms[choice])]]) #choice, reward
                self.av = np.concatenate((self.av, thisAV), axis=0) #add to our action-value memory array


            #calculate the percentage the correct arm is chosen (you can plot this instead of reward)
            percCorrect = 100*(len(self.av[np.where(self.av[:,0] == np.argmax(self.arms))])/len(self.av))


            #calculate the mean reward
            runningMean = np.mean(self.av[:,1])

            plt.scatter(i, runningMean)

        plt.show()


