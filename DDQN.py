import torch
import torch.nn as nn
from herisDispRules import *
from get_fjs import readData
from collections import deque
from torch.distributions import Categorical
from CreatDisjunctiveGraph import creatDisjunctiveGraph
from policyNet import Policy, rewards
import matplotlib.pyplot as plt
import time
import os
import xlwt
import psutil
from caculateTimeCost import CaculateTimeCost
from embeddingModel import get_features_of_node, get_adjacent_matrix


MEMORY_CAPACITY = 10000 #100000                        
BATCH_SIZE = 32                          
# DATASET_BATCH_SIZE = 30
LR = 1e-5                                      
N_STATES = 7                                  
# EPSILON = 0.9                                   # greedy policy
# E_greedy_decrement=0.001                     
GAMMA = 0.9                                     # reward discount
dispatchingRules = ['algorithmSPT', 'algorithmLPT', 'algorithmFOPNR', 'algorithmMORPNR', 'algorithmSR',
                        'algorithmLR', 'randomSolution']

class DDQN(object):
    def __init__(self, policy):
        self.eval_net = policy
        self.target_net = policy
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.EPSILON = 0.9
        self.E_greedy_decrement = 0.001
        self.global_step = 0
        self.update_target_steps = 200

    def choose_action(self, features, adjacent_matrix):
        if np.random.uniform() < self.EPSILON:
            action_pro = self.eval_net(features, adjacent_matrix)
 
            m = Categorical(action_pro)

            action = m.sample().item()
        else:
            action = np.random.randint(N_STATES)
        self.EPSILON = max(0.01, self.EPSILON - self.E_greedy_decrement) 
        return action

    def store_transition(self, s, a, r, s_, done):
        transition = {}
        transition['state'] = deepcopy(s)
        transition['action'] = deepcopy(a)
        transition['reward'] = deepcopy(r)
        transition['nextState'] = deepcopy(s_)
        transition['done'] = deepcopy(done)

        self.memory.append(transition)
        self.memory_counter += 1

    def learn(self):
        print('learning stage')
        if self.global_step % self.update_target_steps == 0:
            self.target_net = self.eval_net
    
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE, replace=False)
        results_eval = []
        results_target = []
        for index in sample_index:
            memory = self.memory[index]
            b_s = memory['state']
            b_a = memory['action']
            b_r = memory['reward']
            b_s_ = memory['nextState']
            b_d = memory['done']
            features = get_features_of_node(b_s)
            adjacent_matrix = get_adjacent_matrix(b_s)
            tmp = self.eval_net(features, adjacent_matrix).detach().numpy().tolist()
            q_eval = tmp[0][b_a]
            q_target = b_r
            if not b_d:
                features = get_features_of_node(b_s_)
                adjacent_matrix = get_adjacent_matrix(b_s_)
                q1 = self.target_net(features, adjacent_matrix).detach().numpy().tolist()[0]
                q_next = self.eval_net(features, adjacent_matrix).detach().numpy().tolist()[0]
                next_action = np.argmax(q_next)
                q_target = b_r + GAMMA * q1[next_action]
                # q_target = q_eval + GAMMA * (b_r + GAMMA * (np.max(q_next) - q_eval))
                # q_target = b_r + GAMMA * np.max(q_next)
            results_eval.append(q_eval)
            results_target.append(q_target)
        loss = self.loss_func(torch.tensor(results_eval), torch.tensor(results_target))
        self.optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        self.optimizer.step()
        self.update_target_steps += 1
        return loss.detach().numpy()

def off_train_DDQN(policy, jobsList, machinesList, maxIter, dispatchingRules):
    method_DDQN = DDQN(policy)

    makespan = []
    TR = []
    for epoch in range(maxIter):
        print('<<<<<<<<<Episode: %s' % epoch)
        # total_reward = 0
        protime = {}
        resultSPT = []
        jobsListCopy = deepcopy(jobsList)
        machinesListCopy = deepcopy(machinesList)
        initState = creatDisjunctiveGraph(jobsList, machinesList)
        state = initState
        protime[0] = {}
        done = False
        while len(resultSPT) != len(jobsListCopy):
            features = get_features_of_node(state)
            adjacent_matrix = get_adjacent_matrix(state)
            action = method_DDQN.choose_action(features, adjacent_matrix)
            algorithmName = dispatchingRules[action]
            current_time = list(protime.keys())[0]
            # timeCost_current = CaculateTimeCost(machinesListCopy)
            jobListToExport, jobsListCopy, protime = eval(algorithmName)(jobsListCopy, machinesListCopy, protime)
            if not protime:
                machineCurrents = []
                for machine in machinesListCopy:
                    machine.currentTime += 1
                    machineCurrents.append(machine.currentTime)
                next_time = max(machineCurrents)
                protime[next_time] = 1
            else:
                next_time = list(protime.keys())[0]
            resultSPT.extend(jobListToExport)

            if len(resultSPT) == len(jobsListCopy):
                done = True
            reward = 1 / CaculateTimeCost(machinesListCopy)
            # reward = rewards(jobListToExport, machinesListCopy, current_time, next_time)
            state_ = creatDisjunctiveGraph(jobsListCopy, machinesListCopy)
            next_state_Infor = state_
            method_DDQN.store_transition(state, action, reward, next_state_Infor, done)
            if method_DDQN.memory_counter > MEMORY_CAPACITY:
                loss = method_DDQN.learn()
                # losses.append(loss)
            state = state_
            # total_reward += reward
        # TR.append(total_reward)

        timeCost = CaculateTimeCost(machinesListCopy)
        makespan.append(timeCost)
     
    # print(makespan)
    return policy, makespan


