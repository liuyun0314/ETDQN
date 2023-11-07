import random

from herisDispRules import *
from GP_rules import *
from get_fjs import readData,read_valid_data
from readInsertJobData import readInsertTaskData,read_insert_data
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from policyNet import *
from DuelingDQN import DuelingDQN_Policy
from randomAction import RA
import time
import os
import xlwt
import scipy.stats as stats
import pandas as pd
from GA_for_DFJSS import GA_DFJSS
from newDispRules import policy_DQN, policy_DQN2, policy_QNGA, policy_DDQN, policy_DuelingDQN, QNGA
from CreatDisjunctiveGraph import creatDisjunctiveGraph
from CreatGraph import plot_graph
from ganttCreator import createGanttChart
from largeScale_instance_generator import instance_generator
from insert_data_generator import insert_instance_generator
from sortedcontainers import SortedDict

max_iter = 20
l=25


def static_dispatch(dispatchingRules,jobsList,macList):
    results = {}
    # max_iter = 20
    count = -1
    algorithmNum = len(dispatchingRules)
    all_timeCost = np.zeros([algorithmNum, max_iter])
    for algorithm in dispatchingRules:
        count += 1
        for epoch in range(max_iter):
            time = {}
            resultSPT = []
            operationsList = deepcopy(jobsList)
            machinesList = deepcopy(macList)
            time[0] = {}
            while len(resultSPT) != len(operationsList):
                print(len(resultSPT))
                jobListToExport, operationsList, time = eval(algorithm)(operationsList, machinesList, time)
                resultSPT.extend(jobListToExport)
                if not time:
                    machine_current_time = []
                    for machine in machinesList:
                        machine.currentTime += 1
                        machine_current_time.append(machine.currentTime)
                    max_machine_time = max(machine_current_time)
                    time[max_machine_time] = {}
            timeCost = CaculateTimeCost(machinesList)
            sorted(resultSPT, key=lambda j: j.startTime)
            if algorithm == 'randomSolution' or algorithm == 'policy_MonteCarlo' or algorithm == 'policy_GA':
                all_timeCost[count, epoch] = timeCost
            else:
                timeCosts = [timeCost] * max_iter
                all_timeCost[count, :max_iter] = timeCosts
                break
    return all_timeCost

def random_instance(jobsList, macsList, insert_job_num, inser_operation):
    all_instances_timeCost = []
    cpu_time = []
    nJob = len(jobsList)

    job_arrival_time = np.random.choice(50, insert_job_num, replace=False)

    count = -1
    rules_sequencing = []
    algorithmNum = len(dispatchingRules)
    all_timeCost = np.zeros([algorithmNum, max_iter])
    for algorithm in dispatchingRules:
        begin_time = time.time()
        count += 1
        print(algorithm)
        for epoch in range(max_iter):
            selected_rules = []
            print(epoch)
            times = {}
            times[0] = 1
            for rescheduleTime in job_arrival_time:
                times[rescheduleTime] = 1
            resultSPT = []
            operationsList = deepcopy(jobsList)
            machinesList = deepcopy(macsList)
            insert_jobs = deepcopy(inser_operation)
            # time[rescheduleTime] = 1
            times = SortedDict(times)
            reschdule_counter = 0
            queue = {}
            for machine in macsList:
                queue[machine.name] = []
            while len(resultSPT) < len(operationsList):
                print('the number of solved operations is %d' % len(resultSPT))
                # print(len(resultSPT))
                current_time = list(times.keys())[0]
                # if current_time == rescheduleTime:
                if current_time in job_arrival_time:
                    operationsList.extend(insert_jobs[reschdule_counter])
                    reschdule_counter += 1
                    # operationsList.extend(insert_jobs)
                    nJob = nJob + 1
                    # print('operationLiat length = %s' %len(operationsList))
                if algorithm == 'GP_rule1' or algorithm == 'GP_rule2':
                    jobListToExport, _, queue, times = eval(algorithm)(operationsList, machinesList, queue, times)
                else:
                    jobListToExport, _, times = eval(algorithm)(operationsList, machinesList, times)
                # jobListToExport, _, times, selected_rule = eval(algorithm)(operationsList, machinesList, times)
                # selected_rules.append(selected_rule)
                if len(jobListToExport) > 0:
                    resultSPT.extend(jobListToExport)
                if not times:
                    machine_current_time = []
                    for machine in machinesList:
                        machine.currentTime += 1
                        machine_current_time.append(machine.currentTime)
                    max_machine_time = max(machine_current_time)
                    times[max_machine_time] = 1
            # rules_sequencing.append(selected_rules)
            timeCost = CaculateTimeCost(machinesList)
            sorted(resultSPT, key=lambda j: j.startTime)

            if algorithm == 'randomSolution' or algorithm == 'policy_DQN' or algorithm == 'policy_QNGA' or algorithm == 'policy_DQN2' or algorithm == 'policy_DDQN' or algorithm == 'RA' or algorithm == 'policy_DuelingDQN':
                all_timeCost[count, epoch] = timeCost
                if epoch == max_iter - 1:
                    end_time = time.time()
                    cost_time = (end_time - begin_time) / max_iter
                    cpu_time.append(cost_time)
            else:
                timeCosts = [timeCost] * max_iter
                all_timeCost[count, :max_iter] = timeCosts
                end_time = time.time()
                cost_time = end_time - begin_time
                cpu_time.append(cost_time)
                break
    all_instances_timeCost.append(all_timeCost)
    return all_instances_timeCost, cpu_time

def dynamic_dispatch_insert(dispatchingRules,jobsList,macList, nJob, insertJobList, insert_num, file_path, dataName):
    results = {}
    rescheduleTime = np.random.randint(0, 30)

    count = -1
    algorithmNum = len(dispatchingRules)
    all_timeCost = np.zeros([algorithmNum, max_iter])
    min_time_cost = np.Inf
    best_operation_results = []
    for algorithm in dispatchingRules:
        # rescheduleTime = np.random.randint(0, 600)
        count += 1
        print(algorithm)
        for epoch in range(max_iter):
            print(epoch)
            time = {}
            resultSPT = []
            operationsList = deepcopy(jobsList)
            machinesList = deepcopy(macList)
            insert_jobs = deepcopy(insertJobList)
            time[0] = 1
            time[rescheduleTime] = 1
            time = SortedDict(time)
            while len(resultSPT) != len(operationsList):
                # print(len(resultSPT))
                current_time = list(time.keys())[0]
                if current_time == rescheduleTime:
                    operationsList.extend(insert_jobs)
                    # operationsList.extend(insert_jobs)
                    nJob = nJob + 1
                    time = SortedDict(time)
                    # print('operationLiat length = %s' %len(operationsList))
                jobListToExport, _, time = eval(algorithm)(operationsList, machinesList, time)
                if len(jobListToExport) > 0:
                    resultSPT.extend(jobListToExport)
                if not time:
                    machine_current_time = []
                    for machine in machinesList:
                        machine.currentTime += 1
                        machine_current_time.append(machine.currentTime)
                    max_machine_time = max(machine_current_time)
                    time[max_machine_time] = 1
            timeCost = CaculateTimeCost(machinesList)
            sorted(resultSPT, key=lambda j: j.startTime)
            if algorithm == 'policy_QNGA':
                if timeCost < min_time_cost:
                    min_time_cost = timeCost
                    best_operation_results = resultSPT
                    best_mahcine_results = machinesList
            if algorithm == 'randomSolution' or algorithm == 'policy_DQN' or algorithm == 'policy_QNGA' or algorithm == 'policy_DQN2':
                all_timeCost[count, epoch] = timeCost
            else:
                timeCosts = [timeCost] * max_iter
                all_timeCost[count, :max_iter] = timeCosts
                break
    # createGanttChart(best_operation_results, best_mahcine_results, file_path, dataName,rescheduleTime)
    return all_timeCost

def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def largerScale_instance_comparison(macsList, jobsList, insert_job_num, inser_operation):
    all_instances_timeCost = []
    all_instances_name = []
    nJob = randon_job_num

    job_arrival_time = np.random.choice(2000, insert_job_num, replace=False)


    count = -1
    algorithmNum = len(dispatchingRules)
    all_timeCost = np.zeros([algorithmNum, max_iter])

    for algorithm in dispatchingRules:
        count += 1
        print(algorithm)
        for epoch in range(max_iter):
            print(epoch)
            time = {}
            time[0] = 1
            for rescheduleTime in job_arrival_time:
                time[rescheduleTime] = 1
            resultSPT = []
            operationsList = deepcopy(jobsList)
            machinesList = deepcopy(macsList)
            insert_jobs = deepcopy(inser_operation)
            # time[rescheduleTime] = 1
            time = SortedDict(time)
            reschdule_counter = 0
            queue = {}
            for machine in macsList:
                queue[machine.name] = []
            while len(resultSPT) != len(operationsList):
                # print(len(resultSPT))
                current_time = list(time.keys())[0]
                # if current_time == rescheduleTime:
                if current_time in job_arrival_time:
                    operationsList.extend(insert_jobs[reschdule_counter])
                    reschdule_counter += 1
                    # operationsList.extend(insert_jobs)
                    nJob = nJob + 1
                    # print('operationLiat length = %s' %len(operationsList))
                if algorithm == 'GP_rule1' or algorithm == 'GP_rule':
                    jobListToExport, _, queue, time = eval(algorithm)(operationsList, machinesList, queue, time)
                else:
                    jobListToExport, _, time = eval(algorithm)(operationsList, machinesList, time)
                if len(jobListToExport) > 0:
                    resultSPT.extend(jobListToExport)
                if not time:
                    machine_current_time = []
                    for machine in machinesList:
                        machine.currentTime += 1
                        machine_current_time.append(machine.currentTime)
                    max_machine_time = max(machine_current_time)
                    time[max_machine_time] = 1
            timeCost = CaculateTimeCost(machinesList)
            sorted(resultSPT, key=lambda j: j.startTime)
            if algorithm == 'randomSolution' or algorithm == 'policy_DQN' or algorithm == 'policy_QNGA' or algorithm == 'policy_DQN2' or algorithm == 'policy_DDQN' or algorithm == 'RA' or algorithm == 'policy_DuelingDQN':
                all_timeCost[count, epoch] = timeCost
            else:
                timeCosts = [timeCost] * max_iter
                all_timeCost[count, :max_iter] = timeCosts
                break
    all_instances_timeCost.append(all_timeCost)
    return all_instances_timeCost, all_instances_name
