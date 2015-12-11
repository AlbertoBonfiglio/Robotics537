#!/usr/bin/env python


#from classes.armedbandit import ArmedBandit
from classes.armedbandit2 import ArmedBandit2, Arm

if __name__ == '__main__':

#    bandit = ArmedBandit(10)
#    bandit.performOneArmRobberyEGreedy(500, 0.1)
#    bandit.performOneArmRobberyEGreedy(500, 0)


    bandit = ArmedBandit2(5)
    bandit.performOneArmRobberyEGreedy(2500, 0)

    #a = bandit.getArm(3)

    #for i in range(0,200):
    #    print('Arm pull ' + str(a.pullArm()))

    pass
