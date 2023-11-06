from sortedcollections import SortedDict
import torch
from copy import deepcopy
from get_fjs import readData
from herisDispRules import *
from GP_rules import *
from policyNet import *
from randomAction import RA
from DuelingDQN import DuelingDQN_Policy
from newDispRules import policy_DQN,policy_QNGA,policy_DQN2,policy_DDQN, policy_DuelingDQN
from caculateTimeCost import CaculateTimeCost
from CreatDisjunctiveGraph import sameJob_op
from newDispRules import policy_MonteCarlo
import time
import os
import xlwt
import scipy.stats as stats
from herisDispRules import *
from get_fjs import readData
from readInsertJobData import readInsertTaskData
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
from insert_data_generator import insert_instance_generator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
max_iter = 30

def recheduleAlgorithm_MachineFaulty(jobsList, machinesList, faulty_Machine, rescheduleTime):
    assignedOperations = {}
    jobListToExport = []
    unScheduleOperations = []
    scheduleOperations = []
    prescheduleOperations = []
    unschedule_operation = {}
    faultyMachine = faulty_Machine.name

        if len(machine.assignedOpera):
            assignedOperations[machine.name] = machine.assignedOpera
            if machine.name != faultyMachine:
                jobListToExport.extend(assignedOperations[machine.name])
            else:
                if assignedOperations[machine.name][-1].endTime <= rescheduleTime:
                    jobListToExport.extend(assignedOperations[machine.name])
                if assignedOperations[machine.name][-1].startTime < rescheduleTime and assignedOperations[machine.name][-1].endTime > rescheduleTime:
                    jobListToExport.extend(assignedOperations[machine.name][:-1])
                    assignedOperations[machine.name][-1].completed = False
                    assignedOperations[machine.name][-1].startTime = 0
                    assignedOperations[machine.name][-1].endTime = 0
                    assignedOperations[machine.name][-1].duration = 0
                    assignedOperations[machine.name][-1].assignedMachine = ''
                    assignedOperations[machine.name][-1].completedRatio = 0
                    if len(assignedOperations[machine.name][-1].machine) == 1:
                        if assignedOperations[machine.name][-1].itinerary not in list(unschedule_operation.keys()):
                            unschedule_operation[assignedOperations[machine.name][-1].itinerary] = []
                            unschedule_operation[assignedOperations[machine.name][-1].itinerary].append(assignedOperations[machine.name][-1])
                        else:
                            unschedule_operation[assignedOperations[machine.name][-1].itinerary].append(assignedOperations[machine.name][-1])

    for job in jobsList:
        if job not in jobListToExport:
            if faultyMachine in job.machine:
                if len(job.machine) == 1:
                    if job.itinerary not in list(unschedule_operation.keys()):
                        unschedule_operation[job.itinerary] = []
                        unschedule_operation[job.itinerary].append(job)
                    else:
                        if job not in unschedule_operation[job.itinerary]:
                            unschedule_operation[job.itinerary].append(job)
                else:
                    prescheduleOperations.append(job)
            else:
                prescheduleOperations.append(job)

    for itinerary, sequeue in unschedule_operation.items():
        sequeue.sort(key=lambda x: x.idOperation)
        unScheduleOperations.extend(sequeue)

    for job in prescheduleOperations:
        if job.itinerary not in list(unschedule_operation.keys()):
            scheduleOperations.append(job)
        else:
            if job.idOperation < unschedule_operation[job.itinerary][0].idOperation:
                scheduleOperations.append(job)

    machinesList = [machine for machine in machinesList if machine.name != faultyMachine]
    scheduleOperations.extend(jobListToExport)
    return jobListToExport, scheduleOperations, machinesList, unScheduleOperations

def dynamic_dispatch_machineFaulty(dispatchingRules, jobsList, macList):
    results = []
    # maxIter = 30
    algorithmNum = len(dispatchingRules)
    all_timeCost = np.zeros([algorithmNum, max_iter])

    rescheduleTime = np.random.randint(1, 300)
    # finish_repair_time = 200
    count = -1
    for algorithm in dispatchingRules:
        random_index = np.random.randint(0, len(macList))
        faultyMachine = macList[random_index]
        count += 1
        print(algorithm)
        for epoch in range(max_iter):
            time = {}
            resultSPT = []
            # operationsList = deepcopy(jobsList)
            machinesList = deepcopy(macList)
            time[0] = 1
            time[rescheduleTime] = 1
            # time[finish_repair_time] = 1
            time = SortedDict(time)
            comingOperasList = deepcopy(jobsList)
            reschedule = False
            while len(resultSPT) != len(comingOperasList):
                print('the number of solved operations is %d' % len(resultSPT))
                if not reschedule:
                    current_time = list(time.keys())[0]
                    if current_time == rescheduleTime:
                        reschedule = True

                        result, comingOperasList, machinesList, Suspended_operations = recheduleAlgorithm_MachineFaulty(comingOperasList, machinesList, faultyMachine, rescheduleTime)
                        resultSPT = deepcopy(result)

                jobListToExport, comingOperasList, time = eval(algorithm)(comingOperasList, machinesList, time)
                leng = len(jobListToExport)
                resultSPT.extend(jobListToExport)
                lengt = len(resultSPT)
                if not time:
                    machine_currentTime = []
                    for machine in machinesList:
                        machine.currentTime += 1
                        machine_currentTime.append(machine.currentTime)
                    maxCurrentTime = max(machine_currentTime)
                    time[maxCurrentTime] = 1
            timeCost = CaculateTimeCost(machinesList)
            if algorithm == 'randomSolution' or algorithm == 'policy_DQN' or algorithm == 'policy_QNGA' or algorithm == 'GA_DFJSS' or algorithm == 'policy_DQN2' or algorithm == 'policy_DDQN' or algorithm == 'RA' or algorithm == 'policy_DuelingDQN':
                all_timeCost[count, epoch] = timeCost
            else:
                all_timeCost[count, :max_iter] = [timeCost] * max_iter
                break
    return all_timeCost

def largerScale_instance_comparison(jobsList, macsList, dispatchingRules):

    algorithmNum = len(dispatchingRules)

    run_times = np.zeros([algorithmNum])

    machine_fault_time = np.random.randint(1, 2000)
    repair_time = np.random.randint(machine_fault_time+1, 2000)
    count = -1
    algorithmNum = len(dispatchingRules)
    all_timeCost = np.zeros([algorithmNum, max_iter])
    for algorithm in dispatchingRules:
        count += 1
        random_index = np.random.randint(1, len(macsList))
        faulty_Machine = macsList[random_index]
        run_time_Set = []
        print(algorithm)
        for epoch in range(max_iter):
            print(epoch)
            process_time = {}
            process_time[0] = 1
            process_time[machine_fault_time] = 1
            process_time[repair_time] = 1
            resultSPT = []
            operationsList = deepcopy(jobsList)
            machinesList = deepcopy(macsList)
            process_time = SortedDict(process_time)
            reschdule_counter = 0
            begin_time = time.time()
            queue = {}
            for machine in macsList:
                queue[machine.name] = []
            while len(resultSPT) < len(operationsList):
                # print(len(resultSPT))
                print('the number of solved operations is %d' % len(resultSPT))
                current_time = list(process_time.keys())[0]
                if current_time == repair_time:
                    machinesList.append(faulty_Machine)
                    candi_oper = []
                    for job_oper in jobsList:
                        if not job_oper.completed:
                            candi_oper.append(job_oper)
                    operationsList = candi_oper
                    queue[faulty_Machine.name] = []
                if current_time == machine_fault_time:
                # if current_time in job_arrival_time:

                    result, operationsList, machinesList, suspended_operations = recheduleAlgorithm_MachineFaulty(operationsList, machinesList, faulty_Machine, machine_fault_time)
                    # operationsList.extend(insert_jobs)
                    resultSPT = deepcopy(result)
                    del queue[faulty_Machine.name]
                    # print('operationLiat length = %s' %len(operationsList))
                if algorithm == 'GP_rule1' or algorithm == 'GP_rule2':
                    jobListToExport, _, queue, process_time = eval(algorithm)(operationsList, machinesList, queue, process_time)
                else:
                    jobListToExport, operationsList, process_time = eval(algorithm)(operationsList, machinesList, process_time)
                if len(jobListToExport) > 0:
                    resultSPT.extend(jobListToExport)
                if not process_time:
                    machine_current_time = []
                    for machine in machinesList:
                        machine.currentTime += 1
                        machine_current_time.append(machine.currentTime)
                    max_machine_time = max(machine_current_time)
                    process_time[max_machine_time] = 1
            end_time = time.time()
            run_time = end_time - begin_time
            run_time_Set.append(run_time)
            timeCost = CaculateTimeCost(machinesList)
            sorted(resultSPT, key=lambda j: j.startTime)
            if algorithm == 'randomSolution' or algorithm == 'policy_DQN' or algorithm == 'policy_QNGA' or algorithm == 'policy_DQN2' or algorithm == 'policy_DDQN' or algorithm == 'RA' or algorithm == 'policy_DuelingDQN':
                all_timeCost[count, epoch] = timeCost
                epoch_run_time = max(run_time_Set)
            else:
                timeCosts = [timeCost] * max_iter
                all_timeCost[count, :max_iter] = timeCosts
                epoch_run_time = run_time_Set[0]
                break
        run_times[count] = epoch_run_time
    return all_timeCost, all_instances_name,run_times

def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    allJobsList = []
    allMachinesList = []
    # dispatchingRules = ['policy_DQN', 'policy_DQN2', 'policy_DDQN', 'policy_DuelingDQN', 'RA', 'policy_QNGA', 'policy_QNGA',
    #                     'randomSolution', 'algorithmSPT', 'algorithmLPT', 'algorithmFOPNR', 'algorithmMORPNR', 'algorithmSR', 'algorithmLR']
    # dispatchingRules = ['randomSolution', 'algorithmSPT', 'algorithmLPT', 'algorithmFOPNR', 'algorithmMORPNR',
    #                     'algorithmSR', 'algorithmLR', 'RA']
    # dispatchingRules = ['policy_DDQN', 'policy_DuelingDQN', 'RA', 'policy_QNGA']
    dispatchingRules = ['GP_rule1', 'GP_rule2']

    # dispatchingRules = ['randomSolution', 'RA']
    # dispatchingRules = ['QNGA']

    randon_job_num = 2000
    random_machine_num = 10
    max_process_time = 100
    max_operation_num = 10

    all_instances_name = []
    valid_machineList = []
    valid_jobsList = []
    machineList, jobsList = instance_generator(randon_job_num, random_machine_num, max_process_time, max_operation_num)
    valid_machineList.append(machineList)
    valid_jobsList.append(jobsList)
    valid_job_num = 1
    # nInsertJob = len(inser_operation)
    dataName = 'random_instance(' + str(randon_job_num) + '*' + str(random_machine_num) + ')'
    all_instances_name.append(dataName)
    for jobsList, machinesList in zip(valid_jobsList, valid_machineList):
            all_timeCost, instance_name, run_times = largerScale_instance_comparison(jobsList, machinesList,
                                                                                         dispatchingRules)

        
        