 #!/usr/bin/env python

import numpy as np
from scipy import stats
import random
import math
import matplotlib.pyplot as plt
import uuid

UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4
NOWHERE = 0
EXIT = 100
UNEXPLORED = -1
EXPLORED  = 0

class Position(object):
    def __init__(self):
        self.x = 0
        self.y = 0
       

    def __eq__(self, other): 
        return self.__dict__ == other.__dict__

    
    def getAvailableActions(self):
        retval = [0,1,2,3,4] # 0 stay in place, 1 = up then right down left 
        if (self.x == 0):
            retval.remove(LEFT)
        if (self.x == 9):
            retval.remove(RIGHT)
        if (self.y == 0):
            retval.remove(UP)
        if (self.y == 4):
            retval.remove(DOWN)
        
        return retval

    def move(self, action):
        newposition = Position()
        newposition.x = self.x
        newposition.y = self.y
        if action == UP:
            newposition.y = self.y -1
        if action == DOWN:
            newposition.y = self.y +1
        if action == RIGHT:
            newposition.x = self.x +1
        if action == LEFT:
            newposition.x = self.x -1
        
        return newposition

    def randomize(self):
        self.x = np.random.randint(0, 8)
        self.y = np.random.randint(0, 4)
        

class Explorer(object):
    def __init__(self):
         
        self.environment_matrix  = np.empty((5, 10), dtype=np.object)
        for row in range(5):
            for column in range(10):
                pos = Position()
                pos.y = row
                pos.x = column
                self.environment_matrix[row][column] = pos
        
        self.current_state = Position()
        self.exit_state = Position() 
        self.exit_state.y = 2
        self.exit_state.x = 9
        self.environment_matrix[2, 9] = self.exit_state
        

    def getRandomStart(self):
        return self.environment_matrix[np.random.randint(4)][np.random.randint(9)]                    
    

    def findPath(self, explorations=500, epsilon=0, alpha=0.2, gamma=0.9):
        retval= []

        self.current_state = self.getRandomStart()
  
        self.agent = qLearner(self.current_state, self.exit_state, epsilon, alpha, gamma)

        for i in range(explorations):
            
            action = self.agent.chooseAction(self.current_state)
            last_state = self.current_state
            move_state = last_state.move(action)
            new_state = self.environment_matrix[move_state.y][move_state.x]

            reward = -1
            if new_state == self.exit_state:
                reward = 100
                       
            self.agent.learn(last_state, action, reward, new_state)
            
            self.current_state = new_state
            self.agent.updateActions(new_state, action)
            
            retval.append(self.agent.getQ(last_state, action))
        
        return retval



class qLearner(object):
    
    def __init__(self, state, exit, epsilon=0.0, alpha=0.2, gamma=0.9):
        self.q = {}
        self.epsilon = epsilon #e-greedy variable 
        self.alpha = alpha #learning rate
        self.gamma = gamma #discount factor
        self.actions = state.getAvailableActions()
        self.exit = exit

    def updateActions(self, state, lastAction):
        self.actions = state.getAvailableActions()
        #if lastAction in self.actions: 
        #    self.actions.remove(lastAction)
        
    

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)


    def learn(self, state1, action1, reward, state2):
        maxQNew = max([self.getQ(state2, a) for a in self.actions]) #max of all possible actions
        self.learnQ(state1, action1, reward, reward + self.gamma * maxQNew)
        
        
    def learnQ(self, state, action, reward, value):
        oldValue = self.q.get((state, action), None)
        if oldValue is None:
            self.q[(state, action)] = reward
        else:
            #if state == self.exit:
            #    self.q[(state, action)] = self.q[(state, action)]
            #else:    
            self.q[(state, action)] += self.alpha * (value - oldValue)


    def chooseAction(self, state):
        if random.random() < self.epsilon:   
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action


    


    