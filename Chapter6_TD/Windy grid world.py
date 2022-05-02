# -*- coding: utf-8 -*-
"""
Created on Sun May  1 11:51:47 2022

@author: Liangying Liu
"""

import numpy as np
import matplotlib as plt

class Windy_grid_world():
    def __init__(self, width, height, start, end, epsilon):
        self.width = width
        self.height = height
        self.start = start
        self.end = end
        self.Q = np.zeros((4, self.width, self.height))     #(z,x,y)
        self.action = np.zeros((self.width, self.height))
        self.epsilon = epsilon
        # 0 Up; 1 down; 2 left; 3 right
        
    def End_state_check(self, state):
        if state == self.end:
            return True
        else:
            return False
        
        
    def State_next(self, state, act):
        row = state[0]
        col = state[1]
        up = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        
        if act == 0:
            return [min(row - up[col], 0), col]
        if act == 1:
            return [min(row + up[col], self.width - 1), col]
        if act == 2:
            return [row + up[col], max(col - 1, 0)]
        if act == 3:
            return [row + up[col], min(col + 1, self.height - 1)]
        
    def Q_next(self, state_next):
        row = state_next[0]
        col = state_next[1]
        if np.random.binomial(1, self.epsilon) == 1:   # 正面朝上的概率是epsilon, 正面为1
            act_next = np.random.choice(range(0,4))
            q_next = self.Q[act_next, row, col]
            return act_next, q_next
        else:
            act_next = np.argmax(self.Q[:,row, col], axis = 0)
            q_next = np.max(self.Q[:,row, col], axis = 0)
            return act_next, q_next
        
    def Q(self, state):
        row = state[0]
        col = state[1]
        if np.random.binomial(1, epsilon) == 1:   # 正面朝上的概率是epsilon, 正面为1
            act = np.random.choice(range(0,4))
            q = self.Q[act, row, col]
            state_next = self.State_next(state, act)
            return act, q, state_next
        else:
            act = np.argmax(self.Q[:,row, col], axis = 0)
            q = np.max(self.Q[:, row, col], axis = 0)
            state_next = self.State_next(state, act)
            return act, q, state_next
        
        '''q = np.zeros(4)
        for act in range(0,4):
            r = -1
            state_next = self.State_next(state, act)
            row = state_next[0]
            col = state_next[1]
            q_next = np.max(self.Q[:,row, col], axis = 0)
            q[act] = r + q_next
        act_max = np.argmax(q)
        q_max = np.max(q)
        self.Q[act_max, state[0], state[1]] = q_max
        
        注意这里是DP与TD的区别，DP是planning，当我们处于当前state时，可以通过模型来比较下一个state哪个最优，此时的action由preplay产生。
        但是TD是没有模型的，当我们处于当前state时，我们只能根据过去的经验replay走哪一步能到达最优的state，此时的action由replay产生，并且我replay的时候               
        所以我这样写的代码是错误的
        这也是为什么TD需要更新，因为过去的经验是不准确的，当你来到下一个state_next，却发现它不是最好的，此时就产生了TD error''' 
     

if __name__ == '__main__':
    start_state = [3,0]
    end_state = [7,3]
    w_grid = Windy_grid_world(10, 7, start_state, end_state, 0.1)
    alpha = 0.5
    
    n = 1000
    for i in range(0,n):
        state = start_state
        while(True):
            if not w_grid.End_state_check(state):
                act, q, state_next = w_grid.Q(state)
                r = -1
                act_next, q_next = w_grid.Q_next(state_next)
                q += alpha * (r + q_next - q)
                w_grid.Q[act,state[0],state[1]] = q
                state = state_next
            else:
                break
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        