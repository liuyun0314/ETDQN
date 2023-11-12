import torch
import os
import xlwt
import time
import numpy as np
import torch.nn as nn
from copy import deepcopy
from collections import deque
import torch.nn.functional as F
from policyNet import Flatten, Policy
from torch.distributions import Categorical
from embeddingModel import get_features_of_node, get_adjacent_matrix
import matplotlib.pyplot as plt
from get_fjs import readData
from herisDispRules import *
from CreatDisjunctiveGraph import creatDisjunctiveGraph
from caculateTimeCost import CaculateTimeCost


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


class DuelingDQN_Policy(nn.Module):
    def __init__(self):
        super().__init__()
        # self.seed = torch.manual_seed(seed)
        self.action_size = 7
        self.fc1_units=64
        p = Policy()
        self.policy = p

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)

        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )

    def forward(self, feartures, edge_index):
        embedding_seq = self.policy.GCN(feartures, edge_index)
        afterMutiAttention = self.policy.multihead_attention(embedding_seq)
        x = F.relu(self.fc1(afterMutiAttention))
        x = F.relu(self.fc2(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        action_pro = F.softmax(q_values, dim=1)

        return action_pro


class DuelingDQN(nn.Module):
    def __init__(self, policy):
        super(DuelingDQN, self).__init__()
        p = DuelingDQN_Policy()
        self.eval_net = p
        self.target_net = p
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
                q_next = self.target_net(features, adjacent_matrix).detach().numpy().tolist()
                q_target = b_r + GAMMA * np.max(q_next)
            results_eval.append(q_eval)
            results_target.append(q_target)
        loss = self.loss_func(torch.tensor(results_eval), torch.tensor(results_target))
        self.optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        self.optimizer.step()
        self.update_target_steps += 1
        return loss.detach().numpy()


def off_train_DuelingDQN(policy, jobsList, machinesList, maxIter, dispatchingRules):
    method_DuelingDQN = DuelingDQN(policy)

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
            action = method_DuelingDQN.choose_action(features, adjacent_matrix)
            algorithmName = dispatchingRules[action]
            current_time = list(protime.keys())[0]
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
            method_DuelingDQN.store_transition(state, action, reward, next_state_Infor, done)
            if method_DuelingDQN.memory_counter > MEMORY_CAPACITY:
                loss = method_DuelingDQN.learn()
                # losses.append(loss)
            state = state_
            # total_reward += reward
        # TR.append(total_reward)

        timeCost = CaculateTimeCost(machinesListCopy)
        makespan.append(timeCost)
      
    # print(makespan)
    return policy, makespan
