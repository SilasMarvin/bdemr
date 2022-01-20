import numpy as np
import random
import sys
import copy 


class BanditsManager:
    def __init__(self, arm_count, epoch_size):
        self.arm_count = arm_count
        self.state = []
        self.allowed_steps = 10
        self.steps = 0
        self.good_arm = None
        self.bandits = []
        self.bandit = None
        self.epoch_size = epoch_size

    def reset_epoch(self):
        self.bandits = []
        contexts = []
        for i in range(self.arm_count):
            bandit = Bandits(self.arm_count)
            while True:
                found = False
                context = np.random.randint(2, size=self.arm_count)
                for c in contexts:
                    if np.array_equal(c, context):
                        found = True
                if not found:
                    break
            contexts.append(context)
            #context = np.zeros(self.arm_count)
            #context[i] = i 
            bandit.state = context
            bandit.good_arm = i
            self.bandits += [copy.deepcopy(bandit) for _ in range(10)]
        random.shuffle(self.bandits)

    def reset(self):
        self.bandit = self.bandits.pop(0)
        return self.bandit.reset()

    def step(self, action):
        return self.bandit.step(action)


class Bandits:
    def __init__(self, arm_count):
        self.arm_count = arm_count
        self.state = []
        self.allowed_steps = 10
        self.steps = 0
        self.good_arm = None

    def reset(self):
        #self.good_arm = np.random.choice(self.arm_count, ())
        #self.state = np.zeros(self.arm_count)
        #self.state[self.good_arm] = 1
        self.steps = 0
        return self.state

    def step(self, action):
        reward = 0
        done = 0
        if action == self.good_arm:
            chance = 0.9
        else:
            chance = 0.1
        if random.random() < chance:
            reward = 1
        else:
            reward = 0 
        self.steps += 1 
        if(self.steps == self.allowed_steps):
            done = 1
        return self.state, reward, done, 0
