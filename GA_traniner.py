import random
import numpy as np
import torch
import psutil
import os
import xlwt
import time
import torch.nn as nn
from policyNet import Policy
from collections import deque
from copy import deepcopy
from policyNet import rewards
from herisDispRules import *
import matplotlib.pyplot as plt
from CreatDisjunctiveGraph import sameJob_op
from torch.distributions import Categorical
from caculateTimeCost import CaculateTimeCost
from embeddingModel import get_features_of_node, get_adjacent_matrix

dispatchingRules = ['algorithmSPT', 'algorithmLPT', 'algorithmFOPNR', 'algorithmMORPNR', 'algorithmSR',
                        'algorithmLR', 'randomSolution']
GAMMA = 0.9

class GA(torch.nn.Module):
    def __init__(self):
       super(GA, self).__init__()
       self.dim = 0
       self.N = 20
       self.pc = 0.8
       self.pm = 0.95
       self.maxFEs = 100
       self.idel_machine = []
       self.candidate_queue = {}

    def caculate_fitness(self, machinesList, X, t, r):
        idel_machine = deepcopy(self.idel_machine)
        incoming_operation = []
        for j in range(self.dim):
            if X[j] >= 0:
                tmp_queue = deepcopy(self.candidate_queue[idel_machine[j].name])
                candidate_operation = tmp_queue[int(X[j])]
                candidate_operation.startTime = t
                candidate_operation.completed = True
                candidate_operation.assignedMachine = idel_machine[j].name
                candidate_operation.duration = candidate_operation.machine[idel_machine[j].name]
                candidate_operation.endTime = candidate_operation.startTime + candidate_operation.duration
                idel_machine[j].assignedOpera.append(candidate_operation)
                idel_machine[j].currentTime = candidate_operation.endTime
                incoming_operation.append(candidate_operation)
        # fitness = CaculateTimeCost(idel_machine)
        allProceTime = CaculateTimeCost(machinesList)
        fitness = r + gamma * (EP / (len(machinesList) * allProceTime))
        return fitness

    def GA_trainer(self, b_s_, b_r):
        self.idel_machine = []
        self.candidate_queue = {}
        jobsList = b_s_['jobs']
        machinesList = b_s_['machines']
        pro_time = b_s_['time']
        resultToExpert = []
        idel_machine_name = []
        corve = []

        step_t = list(pro_time.keys())[0]
        sameJobList = sameJob_op(jobsList)
        for machine in machinesList:
            if machine.currentTime <= step_t:
                self.idel_machine.append(machine)
                idel_machine_name.append(machine.name)

        no_processed_machine = []
        for machine in self.idel_machine:
            self.candidate_queue[machine.name] = []
            for job in jobsList:
                if machine.name in list(job.machine.keys()):
                    if not job.completed:
                        taskID = job.idOperation
                        process = False
                        if taskID > 1:
                            sameJobs = sameJobList[job.itinerary]
                            preOpera = sameJobs[taskID - 2]
                            if preOpera.completed:
                                process = True
                        if taskID == 1:
                            process = True
                        if process:
                            self.candidate_queue[machine.name].append(job)

            if len(self.candidate_queue[machine.name]) == 0:
                no_processed_machine.append(machine)
                self.candidate_queue.pop(machine.name)
        for mac in no_processed_machine:
            self.idel_machine.remove(mac)

        self.dim = len(self.idel_machine)
        # t_timeCost = CaculateTimeCost(machinesList)
        bestf = np.Inf
        bestP = np.zeros(self.dim)
        if len(self.idel_machine) == 1:
            avaliable_machine = self.idel_machine[0].name
            min_operation = 0
            min_PR = self.candidate_queue[avaliable_machine][0].machine[avaliable_machine]
            for job in self.candidate_queue[avaliable_machine][1:]:
                min_operation += 1
                if job.machine[avaliable_machine] < min_PR:
                    min_PR = job.machine[avaliable_machine]
            bestf = min_PR
            bestP[0] = min_operation
        if len(self.idel_machine) > 1:

            X = np.zeros([self.N, self.dim])
            fitness = np.zeros([1,self.N])
            candidate_queue = []
            for i in range(self.N):
                candidate_queue.append(deepcopy(self.candidate_queue))
                # candidate_queue = deepcopy(self.candidate_queue)
                sequeue = []
                for j in range(self.dim):
                    machine = self.idel_machine[j]
                    mac_que = candidate_queue[i][machine.name]
                    index = np.random.randint(0, len(mac_que))
                    if not mac_que[index].completed:
                        X[i, j] = index
                        mac_que[int(X[i, j])].completed = True
                    else:
                        if len(mac_que) == 1:
                            X[i, j] = -1
                        else:
                            no_processed_job = []
                            job_index = 0
                            for job in mac_que:
                                if not job.completed:
                                    no_processed_job.append(job_index)
                                job_index += 1
                            if len(no_processed_job) > 0:
                                index = random.sample(no_processed_job,1)
                                X[i, j] = index[0]  
                                mac_que[int(X[i, j])].completed = True
                            else:
                                X[i, j] = -1

            t = 0
            while t <= self.maxFEs:
                t += 1
                for i in range(self.N):
                    fitness[0,i] = self.caculate_fitness(machinesList, X[i, :], step_t, b_r)
                    if fitness[0,i] < bestf:
                        bestf = fitness[0,i]
                        bestP = X[i, :]
                # --------------------selection----------------------- 
                total_fitness = sum(fitness)
                selection_probs = [f / total_fitness for f in fitness]
                cumulative_probs = [sum(selection_probs[:i+1]) for i in range(len(selection_probs))]

                selected_individuals = []
                for _ in range(len(X)):
                    rand_num = random.random()
                    for i in range(len(cumulative_probs)):
                        if rand_num <= cumulative_probs[i]:
                            selected_individuals.append(X[i])
                            break    
                X = selected_individuals
                
                # ----------------------crossover--------------------------
                newX = []
                for i in range(self.N):
                    if random.random() <= self.pc:
                        s = random.sample(range(self.N), 2)
                        father = X[s[0], :]
                        mother = X[s[1], :]
                        point = np.random.randint(0,self.dim)
                        newX[s[0], point:] = mother[point:]
                        newX[s[1], :point] = father[:point]
                X = newX

                # ----------------------Mutation--------------------------
                for i in range(self.N):
                    if random.random() <= self.pm:
                        index = np.random.randint(0, self.dim)
                        mac = self.idel_machine[index]
                        mac_queue = candidate_queue[i][mac.name]

                        random_index = np.random.randint(0, len(mac_queue))
                        if random_index != newX[i, index]:
                            if not mac_queue[random_index].completed:
                                mac_queue[int(newX[i, index])].completed = False
                                newX[i, index] = random_index
                                mac_queue[random_index].completed = True
                X = newX
                corve.append(bestf)

        for j in range(self.dim):
            if bestP[j] >= 0:
                idel_mac = self.idel_machine[j]
                idel_mac_queue = self.candidate_queue[idel_mac.name]
                candidate_operation = idel_mac_queue[int(bestP[j])]
                candidate_operation.startTime = step_t
                candidate_operation.completed = True
                candidate_operation.assignedMachine = idel_mac.name
                candidate_operation.duration = candidate_operation.machine[idel_mac.name]
                candidate_operation.endTime = candidate_operation.startTime + candidate_operation.duration
                idel_mac.assignedOpera.append(candidate_operation)
                idel_mac.currentTime = candidate_operation.endTime
                resultToExpert.append(candidate_operation)
                if candidate_operation.endTime not in list(pro_time.keys()):
                    pro_time[candidate_operation.endTime] = 1
                else:
                    pro_time[candidate_operation.endTime] += 1
        del pro_time[step_t]
        pro_time = SortedDict(pro_time)
        fit = CaculateTimeCost(machinesList)

        return 1/fit, resultToExpert, jobsList, machinesList, pro_time

    def forward(self, b_s_):
        reward,resultToExpert,jobsList, machinesList,pro_time = self.GA_trainer(b_s_)
        return reward,resultToExpert,jobsList, machinesList,pro_time   

class QNGA(object):

    def __init__(self, policy, GA_selection): 
        self.eval_net = policy
        # self.target_net = deepcopy(policy)
        self.GA_trainer = GA_selection
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.MEMORY_CAPACITY = 1000
        self.BATCH_SIZE = 64
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=1e-5  ) 
        self.loss_func = nn.MSELoss() 
        self.memory = deque(maxlen=10000)
        self.GA_memory = deque(maxlen=10000)
        self.N_STATES = 7
        self.EPSILON = 0.9  
        self.E_greedy_decrement = 0.001 

    # exploartion
    def choose_action(self, features, adjacent_matrix):
        if np.random.uniform() < self.EPSILON:
            action_pro = self.eval_net(features, adjacent_matrix)
            m = Categorical(action_pro)
            action = m.sample().item()
        else:
            action = np.random.randint(self.N_STATES)
        self.EPSILON = max(0.01, self.EPSILON - self.E_greedy_decrement)  
        return action

    def store_transition(self, s, c_s, a, r, s_):
        transition = {}
        transition['state'] = deepcopy(s)
        transition['action'] = deepcopy(a)
        transition['reward'] = deepcopy(r)
        transition['nextState'] = deepcopy(s_)
        transition['current_state'] = deepcopy(c_s)
        self.memory.append(transition)
        self.memory_counter += 1

    def store_state(self,current_state, reward):
        station = {}
        station['state'] = deepcopy(current_state)
        station['reward'] = deepcopy(reward)
        self.memory.append(station)
        self.memory_counter += 1

    def store_state_reward(self, s, r):
        state_reward = {}
        state_reward['state'] = deepcopy(s)
        state_reward['GA_reward'] = deepcopy(r)
        self.GA_memory.append(state_reward)

    def learn(self):
        print('learning stage')
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE, replace=False)
        results_eval = []
        results_target = []
        for index in sample_index:
            memory = self.memory[index]
            b_r = memory['reward']
            s_ = memory['current_state']
            # s_ = memory['nextState']
            # s = b_s_['state']
            process = True
            for m in self.GA_memory:
                if s_ == m['state']:
                    target_reward = m['GA_reward']
                    process = False
                    break
            if process:
                target_reward, _, _, _, _ = self.GA_trainer(s_, b_r)
                self.store_state_reward(s_, target_reward)
            results_eval.append(b_r)
            results_target.append(target_reward)
        loss = self.loss_func(torch.tensor(results_eval), torch.tensor(results_target))
        self.optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        self.optimizer.step()
        return loss.detach().numpy()

def off_train_QNGA(policy, jobsList, machinesList, maxIter):
    GA_trainer = GA()
    method_QNGA = QNGA(policy, GA_trainer)
    makespan = []
    for epoch in range(maxIter):
        print('<<<<<<<<<Episode: %s' % epoch)
        total_reward = 0
        protime = {}
        actions = []
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
            action = method_QNGA.choose_action(features,adjacent_matrix)
            algorithmName = dispatchingRules[action]
            current_time = list(protime.keys())[0]
            current_state = {}
            current_state['jobs'] = deepcopy(jobsListCopy)
            current_state['machines'] = deepcopy(machinesListCopy)
            current_state['time'] = deepcopy(protime)

            jobListToExport, jobsListCopy, protime = eval(algorithmName)(jobsListCopy, machinesListCopy, protime)
            if not protime:
                machineCurrents = []
                for machine in machinesListCopy:
                    machine.currentTime += 1
                    machineCurrents.append(machine.currentTime)
                next_time = max(machineCurrents)
                if next_time not in list(protime.keys()):
                    protime[next_time] = 1
                else:
                    protime[next_time] += 1
            else:
                next_time = list(protime.keys())[0]
            resultSPT.extend(jobListToExport)

            if len(resultSPT) == len(jobsListCopy):
                done = True
            reward = 1 / CaculateTimeCost(machinesListCopy)
            state_ = creatDisjunctiveGraph(jobsListCopy, machinesListCopy)
            next_state_Infor = {}
            next_state_Infor['jobs'] = jobsListCopy
            next_state_Infor['machines'] = machinesListCopy
            next_state_Infor['time'] = protime
            method_QNGA.store_state_reward(current_state, reward)
            method_QNGA.store_transition(state,current_state,action, reward,next_state_Infor)
            if method_QNGA.memory_counter > method_QNGA.MEMORY_CAPACITY:
                loss = method_QNGA.learn()
            state = state_

        timeCost = CaculateTimeCost(machinesListCopy)
        makespan.append(timeCost)

    return policy,makespan
