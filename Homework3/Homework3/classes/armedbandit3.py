#!/usr/bin/env python

import numpy as np
from scipy import stats
import random
import math

UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4
NOWHERE = 0
EXIT = 100
UNEXPLORED = -1
EXPLORED  = 0
class Position(object):
    x = 0
    y = 0

    def __eq__(self, other): 
        return self.__dict__ == other.__dict__


class Arm(object):
    pullCount = 0
    current_value = 0.0
    iterations = 20
    position_matrix = np.full((5, 10), UNEXPLORED)  # Y,X
    current_position = Position()
    exit_position = Position()

    
    def __init__(self):
        self.position_matrix[2, 9] = EXIT
        self.exit_position.y = 2
        self.exit_position.x = 9

            
    def pullArm(self, iterations=10):
        try:
            self.pullCount += 1
            n = self.pullCount
 
            self.iterations = iterations
 
            self.position_matrix = np.full((5, 10), UNEXPLORED)
            self.position_matrix[2, 9] = EXIT
        
            self.current_position.x = np.random.randint(0, 8)
            self.current_position.y = np.random.randint(0, 4)
         
            
            value = self.current_value
            self.current_value = ((n -1)/float(n)) * value + (1 / float(n)) * self.getReward(self.current_position)
       

#            print self.position_matrix[0]
#            print self.position_matrix[1]
#            print self.position_matrix[2]
##            print self.position_matrix[3]
#            print self.position_matrix[4]
#            print '-------------------------------------------------------------------'

            return self.current_value
        except Exception as ex:
            print('PullArm Exception: ' + ex.message)




    def getReward(self, position): 
        try:
            # basically we pick a random possible move from the current position out of 5 positions
            # and we sum the reward
            reward = 0
            for i in range(self.iterations):
                moves = self.getAvailableMoves(position)
                self.getNewPosition(moves, position)            
        
                # update the position matrix to show where the exploration led
                self.position_matrix[position.y, position.x] = EXPLORED
        
                stepReward = self.isExitFound(position)
                reward = reward + stepReward
                if stepReward == EXIT: break 

            return reward
        except Exception as ex:
            print('getReward Exception: ' + ex.message)


    def getAvailableMoves(self, position):
        retval = [0,1,2,3,4] # 0 stay in place, 1 = up then right down left 
        if (position.x == 0):
            retval.remove(LEFT)
        if (position.x == 9):
            retval.remove(RIGHT)
        if (position.y == 0):
            retval.remove(UP)
        if (position.y == 4):
            retval.remove(DOWN)
        
        return retval


    def getNewPosition(self, moves, position):
        move = np.random.choice(moves)
        if move == UP:
            position.y = position.y -1
        if move == DOWN:
            position.y = position.y +1
        if move == RIGHT:
            position.x = position.x +1
        if move == LEFT:
            position.x = position.x -1
            

    def isExitFound(self, position):
        retval = UNEXPLORED
        if (position == self.exit_position):
            retval = EXIT
        return retval
                   
    


class ArmedBandit3(object):
    arms = []
    #initialize memory array; has 1 row defaulted to random action index
    av = []

    def __init__(self, arms=5, startValue=0):
        try:
            self.startValue = startValue

            for count in range(arms):
                obj = Arm()
                self.arms.append(obj)
 
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
        try:           
            self.av =  np.array([0, self.startValue]).reshape(1, 2)
            self.av =  np.concatenate((self.av, [[1, self.startValue]]), axis=0)
            self.av =  np.concatenate((self.av, [[2, self.startValue]]), axis=0)
            self.av =  np.concatenate((self.av, [[3, self.startValue]]), axis=0)
            self.av =  np.concatenate((self.av, [[4, self.startValue]]), axis=0)

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



if __name__ == '__main__':

    arm = Arm()
    print(arm.pullArm())
