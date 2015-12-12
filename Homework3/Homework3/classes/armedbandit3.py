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
    position_matrix = np.full((5, 10),-1)
    

    def __init__(self):
        self.position_matrix[0,9] = 100
        print self.position_matrix[0,9]
        
            
    def pullArm(self, iterations=10):
        self.pullCount += 1
        n = self.pullCount
 
        self.iterations = iterations
 
        value = self.current_value
        self.current_value = ((n -1)/float(n)) * value + (1 / float(n)) * self.getReward()
        
        return self.current_value


    def getReward(self): 
        # basically we pick a random possible move from the current position out of 5  n times
        # and we sum the reward
        reward = 0
        for i in range(self.iterations):
            step = np.random.randint(0 ,5)

            reward = reward + np.random.normal(self.mu, self.sigma)
        return reward/self.iterations


class ArmedBandit3(object):
    """description of class"""


if __name__ == '__main__':

    arm = Arm()
    arm.pullArm()
