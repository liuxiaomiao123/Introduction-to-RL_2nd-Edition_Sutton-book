# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 21:59:18 2022

@author: 90542
"""

import numpy as np
import pandas as pd
import matplotlib as plt

class Random_walk():
    def __init__(self, n_states, prob):
        self.n_states = n_states 
        self.prob = prob
        self.count_one_episode = np.zeros(self.n_states)
        self.path = []
        #self.values = np.zeros(self.n_states)
        self.values = np.full(self.n_states, 0.5)
        self.values[0] = self.values[self.n_states - 1] = 0   # 所有终止态的value都为0
    
    def reset_MC(self):
        #self.values = np.zeros(self.n_states)
        self.values = np.full(self.n_states, 0.5)
        self.values[0] = self.values[self.n_states - 1] = 0
        self.count_one_episode = np.zeros(self.n_states)
        self.path = []
        
    def reset_TD(self):   # TD中values是互相依赖的，不能在每一幕中初始化
        self.count_one_episode = np.zeros(self.n_states)
        self.path = []
        
    def End_state_check(self, state):
        if state == self.n_states - 1 or state == 0:
            return True
        else:
            return False
        
    def Action(self, state):
        #if not self.End_state_check(state):   # 类中方法调用类中其他方法的其中一种写法
        return state - 1, state + 1
        
    def Value_update_DP(self, next_states):
        v = 0
        for next_state in next_states:
            if next_state == self.n_states - 1:
                r = 1
            else:
                r = 0
            v += self.prob * (r + self.values[next_state])
        return v
   
    def Value_update_TD(self, alpha, discount, state, next_state):
        v = 0
        if next_state == self.n_states - 1:
            r = 1
        else:
            r = 0
        v = r + discount * self.values[next_state]
        self.values[state] += alpha * (v - self.values[state])
        
    
    def Return_update_MC(self, discount, state, path):
        G = np.zeros(self.n_states)
        if state == self.n_states - 1:
            path.append(state)    #将终止态加入路径
            G[state] = 1
            T = len(path)
            for i in range(T-2, -1, -1):
                s = path[i]
                s_next = path[i+1]
                G[s] = 0 + discount * G[s_next]    # 注意，不需要做判断，因为从后往前计算时，相当于就地更新，最后的就是我们一开始的第一个态
        return G
    


def DP():   # policy_evaluation_in_place
    rw = Random_walk(7, 0.5)
    while(True):
        values_old = rw.values.copy()
        for state in range(rw.n_states):
            if not rw.End_state_check(state):
                next_states = rw.Action(state)   # 自动成为数组接收来自函数的多个返回值
                rw.values[state] = rw.Value_update_DP(next_states)
        delta = np.abs(values_old - rw.values)
        
        if np.max(delta) < 1e-4:
            print("DP: ")
            print(np.around(rw.values,2))
            break
        
def MC(n, discount, alpha):
    rw = Random_walk(7, 0.5)
    count_across_episodes = np.zeros(7)
    value_across_episodes = np.zeros(7)
    for i in range(n):
        rw.reset_MC()  # MC不像DP或者TD, 其状态不依赖于上一轮迭代得到的状态，所以每一幕都要初始化
        state = np.random.randint(1, 5)    # 将初始状态随机化
        while(True):
            if not rw.End_state_check(state):   # 如果state是终止态，则不需要计数
                rw.path.append(state)
                rw.count_one_episode[state] += 1    
                next_state = np.random.choice(np.array(rw.Action(state)))   
                state = next_state
            else:
                G = rw.Return_update_MC(discount, state, rw.path)
                break
        count_across_episodes += np.where(rw.count_one_episode > 0, 1, 0)  # 减少for循环的写法
        for j in range(7):
            if rw.count_one_episode[j] > 0:
                #value_across_episodes[j] += (G[j] - value_across_episodes[j]) / count_across_episodes[j]
                value_across_episodes[j] += alpha * (G[j] - value_across_episodes[j])
    
        # value_across_episodes += (G - value_across_episodes) / count_across_episodes    
        # 一定要注意, value不能用以数组为单位进行更新。因为如果某一幕中没有出现上一幕的状态，那么这些状态的value应该是不变的。但是以数组为单位就意味着它们也被更新了。
    
    print("MC: ")
    print(np.around(value_across_episodes,2))
    
    
def TD(n, discount, alpha):
    rw = Random_walk(7, 0.5)
    for i in range(n):      # TD具有DP的性质，需要在每一轮迭代的基础上继续更新，所以不能初始化
        rw.reset_TD()
        state = np.random.randint(1,5)
        while(True):
            if not rw.End_state_check(state):  
                rw.path.append(state)
                next_state = np.random.choice(np.array(rw.Action(state)))
                rw.Value_update_TD(alpha, discount, state, next_state)
                state = next_state
            else:
                break
    print("TD: ")
    print(np.round(rw.values, 2))    # 这里的2千万不能少
    
    
if __name__ == '__main__':
    DP()
    MC(100, 1, 0.01)    # 注意discount与alpha的区别
    TD(100, 1, 0.01)