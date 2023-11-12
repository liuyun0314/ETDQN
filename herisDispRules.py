import random
import numpy as np
from copy import deepcopy
from get_fjs import readData
# from sortedcollections import SortedDict
from sortedcontainers import SortedDict
from parseData import parseData
from algorithms import prepareJobs
from ganttCreator import createGanttChart
from caculateTimeCost import CaculateTimeCost

from readInsertJobData import readInsertTaskData
from largeScale_instance_generator import instance_generator
from CreatDisjunctiveGraph import sameJob_op,creatDisjunctiveGraph, disjunctive_ops,plot_graph


def algorithmSPT(aJobsList, machinesList, time):
    '''
    shortest processing time
    operation with shortest processing time that can be processed
    :param aJobsList:
    :return:
    '''
    # print('algorithmSPT')
    # global machinesList
    waitingOperations = {}
    jobsListToExport = []
    sameTaskList = sameJob_op(aJobsList)

    machinesName = [machine.name for machine in machinesList]
    currentTime = list(time.keys())[0]

    if currentTime == 0:  
        # initialize machine times and get first waiting operation for each machine
        for machine in machinesList:
            waitingOperations[machine.name] = []
            for job in aJobsList:
                if job.idOperation == 1 and machine.name in list(job.machine.keys()):
                    if len(job.machine) == 1:
                        waitingOperations[machine.name].append(job)
                    else:
                        workedMachines = job.machine
                        minTimeMachine = machine.name
                        minPRT = workedMachines[minTimeMachine]
                        for mac, PRT in workedMachines.items():
                            if PRT < minPRT:
                                minPRT = PRT
                                minTimeMachine = mac
                        if minTimeMachine == machine.name:
                            waitingOperations[machine.name].append(job)
        for keyMach, operation in waitingOperations.items():
            if len(operation):
                if len(operation) == 1:
                    candidate = operation[0]
                if len(operation) > 1:
                    candidate = operation[0]
                    candidatePRT = candidate.machine[keyMach]
                    for task in operation:
                        if task.machine[keyMach] < candidatePRT:
                            candidate = task
                            candidatePRT = task.machine[keyMach]
                if candidate not in jobsListToExport:
                    candidate.startTime = currentTime
                    candidate.completed = True
                    candidate.assignedMachine = keyMach
                    candidate.duration = candidate.machine[keyMach]
                    candidate.endTime = candidate.startTime + candidate.duration
                    for machine in machinesList:
                        if machine.name == keyMach:
                            machine.currentTime = candidate.endTime
                            machine.assignedOpera.append(candidate)
                            # time[machine.currentTime] = {}
                            if machine.currentTime not in list(time.keys()):
                                time[machine.currentTime] = 1
                            else:
                                time[machine.currentTime] += 1
                            break
                    jobsListToExport.append(candidate)
        del time[0]

    else:
        # print(1)
        idelMachine = []
        t = list(time.keys())[0]
        for machine in machinesList:
            if machine.currentTime <= float(t):
                idelMachine.append(machine)
        for mac in idelMachine:
            waitingOperations[mac.name] = []
            for task in aJobsList:
                if not task.completed and mac.name in list(task.machine.keys()):
                    taskID = task.idOperation
                    if taskID == 1:
                        waitingOperations[mac.name].append(task)
                    else:
                        sameItineraryTasks = sameTaskList[task.itinerary]
                        preOpera = sameItineraryTasks[taskID-2]
                        if preOpera.completed and preOpera.endTime <= float(t):
                            waitingOperations[mac.name].append(task)
        for macName, tasks in waitingOperations.items():
            macIndex = machinesName.index(macName)
            mac = machinesList[macIndex]
            process = False
            if len(tasks) == 1:
                if tasks[0] not in jobsListToExport:
                    tasks[0].startTime = float(t)
                    tasks[0].completed = True
                    tasks[0].assignedMachine = macName
                    tasks[0].duration = tasks[0].machine[macName]
                    tasks[0].endTime = tasks[0].startTime + tasks[0].duration
                    mac.currentTime = tasks[0].endTime
                    mac.assignedOpera.append(tasks[0])
                    # time[mac.currentTime] = {}
                    if mac.currentTime not in list(time.keys()):
                        time[mac.currentTime] = 1
                    else:
                        time[mac.currentTime] += 1
                    jobsListToExport.append(tasks[0])
            if len(tasks) > 1:
                candidate = tasks[0]
                candidatePRT = candidate.machine[macName]
                for task in tasks:
                    if task.machine[macName] < candidatePRT:
                        candidate = task
                        candidatePRT = task.machine[macName]
                if candidate not in jobsListToExport:
                    # if process and theOpera not in jobsListToExport:
                    candidate.startTime = float(t)
                    candidate.completed = True
                    candidate.assignedMachine = macName
                    candidate.duration = candidate.machine[macName]
                    candidate.endTime = candidate.startTime + candidate.duration
                    mac.assignedOpera.append(candidate)
                    mac.currentTime = candidate.endTime
                    # time[mac.currentTime] = {}
                    if mac.currentTime not in list(time.keys()):
                        time[mac.currentTime] = 1
                    else:
                        time[mac.currentTime] += 1
                    jobsListToExport.append(candidate)
        del time[t]
    time = SortedDict(time)
    aJobsList, _ = computingCompletedRatio(aJobsList)
    return jobsListToExport, aJobsList, time

def algorithmLPT(aJobsList, machinesList, time):
    '''
    longest processing time
    operation with longest processing time that can be processed
    :param aJobsList:
    :return:
    '''
    # print('algorithmSPT')
    # global machinesList
    waitingOperations = {}
    jobsListToExport = []
    sameTaskList = sameJob_op(aJobsList)

    machinesName = [machine.name for machine in machinesList]
    currentTime = list(time.keys())[0]

    if currentTime == 0:  
        # initialize machine times and get first waiting operation for each machine
        for machine in machinesList:
            waitingOperations[machine.name] = []
            for job in aJobsList:
                if job.idOperation == 1 and machine.name in list(job.machine.keys()):
                    if len(job.machine) == 1:
                        waitingOperations[machine.name].append(job)
                    else:
                        workedMachines = job.machine
                        maxTimeMachine = machine.name
                        maxPRT = workedMachines[maxTimeMachine]
                        for mac, PRT in workedMachines.items():
                            if PRT > maxPRT:
                                maxPRT = PRT
                                maxTimeMachine = mac
                        if maxTimeMachine == machine.name:
                            waitingOperations[machine.name].append(job)
        for keyMach, operation in waitingOperations.items():
            if len(operation):
                if len(operation) == 1:
                    candidate = operation[0]
                if len(operation) > 1:
                    candidate = operation[0]
                    candidatePRT = candidate.machine[keyMach]
                    for task in operation:
                        if task.machine[keyMach] > candidatePRT:
                            candidate = task
                            candidatePRT = task.machine[keyMach]
                if candidate not in jobsListToExport:
                    candidate.startTime = currentTime
                    candidate.completed = True
                    candidate.assignedMachine = keyMach
                    candidate.duration = candidate.machine[keyMach]
                    candidate.endTime = candidate.startTime + candidate.duration
                    for machine in machinesList:
                        if machine.name == keyMach:
                            machine.currentTime = candidate.endTime
                            machine.assignedOpera.append(candidate)
                            # time[machine.currentTime] = {}
                            if machine.currentTime not in list(time.keys()):
                                time[machine.currentTime] = 1
                            else:
                                time[machine.currentTime] += 1
                            break
                    jobsListToExport.append(candidate)
        del time[0]

    else:
        # print(1)
        idelMachine = []
        t = list(time.keys())[0]
        for machine in machinesList:
            if machine.currentTime <= float(t):
                idelMachine.append(machine)
        for mac in idelMachine:
            waitingOperations[mac.name] = []
            for task in aJobsList:
                if not task.completed and mac.name in list(task.machine.keys()):
                    taskID = task.idOperation
                    if taskID == 1:
                        waitingOperations[mac.name].append(task)
                    else:
                        sameItineraryTasks = sameTaskList[task.itinerary]
                        preOpera = sameItineraryTasks[taskID-2]
                        if preOpera.completed and preOpera.endTime <= float(t):
                            waitingOperations[mac.name].append(task)
        for macName, tasks in waitingOperations.items():
            macIndex = machinesName.index(macName)
            mac = machinesList[macIndex]
            process = False
            if len(tasks) == 1:
                if tasks[0] not in jobsListToExport:
                    tasks[0].startTime = float(t)
                    tasks[0].completed = True
                    tasks[0].assignedMachine = macName
                    tasks[0].duration = tasks[0].machine[macName]
                    tasks[0].endTime = tasks[0].startTime + tasks[0].duration
                    mac.currentTime = tasks[0].endTime
                    mac.assignedOpera.append(tasks[0])
                    # time[mac.currentTime] = {}
                    if mac.currentTime not in list(time.keys()):
                        time[mac.currentTime] = 1
                    else:
                        time[mac.currentTime] += 1
                    jobsListToExport.append(tasks[0])
            if len(tasks) > 1:
                candidate = tasks[0]
                candidatePRT = candidate.machine[macName]
                for task in tasks:
                    if task.machine[macName] > candidatePRT:
                        candidate = task
                        candidatePRT = task.machine[macName]
                if candidate not in jobsListToExport:
                    # if process and theOpera not in jobsListToExport:
                    candidate.startTime = float(t)
                    candidate.completed = True
                    candidate.assignedMachine = macName
                    candidate.duration = candidate.machine[macName]
                    candidate.endTime = candidate.startTime + candidate.duration
                    mac.assignedOpera.append(candidate)
                    mac.currentTime = candidate.endTime
                    # time[mac.currentTime] = {}
                    if mac.currentTime not in list(time.keys()):
                        time[mac.currentTime] = 1
                    else:
                        time[mac.currentTime] += 1
                    jobsListToExport.append(candidate)
        del time[t]
    time = SortedDict(time)
    aJobsList, _ = computingCompletedRatio(aJobsList)
    return jobsListToExport, aJobsList, time

def algorithmFOPNR(aJobsList, machinesList, time):
    '''
    Fewest Operation Remaining
    :param aJobsList:
    :param time:
    :return:
    '''
    # print('algorithmFOPNR')
    waitingOperations = []
    machinesName = []
    jobsListToExport = []
    sameTaskList = sameJob_op(aJobsList)

    for machine in machinesList:
        machinesName.append(machine.name)

    currentTime = list(time.keys())[0]
    # if not time:
    if currentTime == 0:
        for machine in machinesList:
            for job in aJobsList:
                if not job.completed and job.idOperation == 1 and machine.name in list(job.machine.keys()):
                    if len(job.machine) == 1:
                        waitingOperations.append(job)
                if len(waitingOperations):
                    if len(waitingOperations) == 1:
                        candidate = waitingOperations[0]
                    if len(waitingOperations) > 1:
                        candidate = waitingOperations[0]
                        itinerary = candidate.itinerary
                        same_list = sameTaskList[itinerary]
                        min_same_itinerary_num = len(same_list)
                        # candidatePRT = candidate.machine[machine.name]
                        for task in waitingOperations:
                            itinerary = task.itinerary
                            same_list = sameTaskList[itinerary]
                            same_itinerary_num = len(same_list)
                            if same_itinerary_num < min_same_itinerary_num:
                                candidate = task
                                min_same_itinerary_num = same_itinerary_num
                                # candidatePRT = task.machine[machine.name]
                    if candidate not in jobsListToExport:
                        candidate.startTime = currentTime
                        candidate.completed = True
                        candidate.assignedMachine = machine.name
                        candidate.duration = candidate.machine[machine.name]
                        candidate.endTime = candidate.startTime + candidate.duration
                        machine.currentTime = candidate.endTime
                        machine.assignedOpera.append(candidate)
                        # time[machine.currentTime] = {}
                        if machine.currentTime not in list(time.keys()):
                            time[machine.currentTime] = 1
                        else:
                            time[machine.currentTime] += 1
                        jobsListToExport.append(candidate)
        del time[0]
    else:
        idelMachines = []
        t = list(time.keys())[0]
        for machine in machinesList:
            if float(t) >= machine.currentTime:
                idelMachines.append(machine)
        for mac in idelMachines:
            macName = mac.name
            waitingOperations = []
            for task in aJobsList:
                if macName in list(task.machine.keys()) and not task.completed:
                    if task.idOperation == 1:
                        waitingOperations.append(task)  
                    else:
                        itinerary = task.itinerary
                        job_id = task.idOperation
                        operList = sameTaskList[itinerary]
                        prev_operatoin = operList[job_id-2]
                        if prev_operatoin.completed and prev_operatoin.endTime <= float(t):
                            waitingOperations.append(task)
            if len(waitingOperations):
                itinerary = waitingOperations[0].itinerary
                job_id = waitingOperations[0].idOperation
                same_job_list = sameTaskList[itinerary]
                last_job_id = same_job_list[-1].idOperation
                min_no_process_num = last_job_id - job_id + 1
                incomingOperation = waitingOperations[0]
                for task in waitingOperations[1:]:
                    itinerary = task.itinerary
                    job_id = task.idOperation
                    same_job_list = sameTaskList[itinerary]
                    last_job_id = same_job_list[-1].idOperation
                    no_process_num = last_job_id - job_id + 1
                    if no_process_num < min_no_process_num:
                        min_no_process_num = no_process_num
                        incomingOperation = task
                if incomingOperation not in jobsListToExport:
                    incomingOperation.startTime = float(t)
                    incomingOperation.completed = True
                    incomingOperation.duration = incomingOperation.machine[mac.name]
                    incomingOperation.assignedMachine = mac.name
                    mac.assignedOpera.append(incomingOperation)
                    mac.currentTime = incomingOperation.getEndTime()
                    # time[mac.currentTime] = {}
                    if mac.currentTime not in list(time.keys()):
                        time[mac.currentTime] = 1
                    else:
                        time[mac.currentTime] += 1
                    jobsListToExport.append(incomingOperation)
        del time[t]
        time = SortedDict(time)
        aJobsList,_ = computingCompletedRatio(aJobsList)
    return jobsListToExport,aJobsList,time

def algorithmMORPNR(aJobsList, machinesList, time):
    '''
    Most Operation Remaining
    :param aJobsList:
    :param time:
    :return:
    '''
    # print('algorithmMORPNR')
    waitingOperations = []
    machinesName = []
    jobsListToExport = []

    for machine in machinesList:
        machinesName.append(machine.name)

    sameJobOpera = sameJob_op(aJobsList)

    currentTime = list(time.keys())[0]
    # if not time:
    if currentTime == 0:
        for machine in machinesList:
            # waitingOperations[machine.name] = []
            for job in aJobsList:
                if not job.completed and job.idOperation == 1 and machine.name in list(job.machine.keys()):
                    if len(job.machine) == 1:
                        waitingOperations.append(job)
            if len(waitingOperations):
                if len(waitingOperations) == 1:
                    candidate = waitingOperations[0]
           
                if len(waitingOperations) > 1:
                    candidate = waitingOperations[0]
                    itinerary = candidate.itinerary
                    same_list = sameJobOpera[itinerary]
                    max_same_itinerary_num = len(same_list)
                    # candidatePRT = candidate.machine[keyMach]
                    for task in waitingOperations:
                        itinerary = task.itinerary
                        same_list = sameJobOpera[itinerary]
                        same_itinerary_num = len(same_list)
                        if same_itinerary_num > max_same_itinerary_num:
                            candidate = task
                            max_same_itinerary_num = same_itinerary_num
                        # if task.machine[keyMach] < candidatePRT:
                        #     candidate = task
                        #     candidatePRT = task.machine[keyMach]
                if candidate not in jobsListToExport:
                    candidate.startTime = currentTime
                    candidate.completed = True
                    candidate.assignedMachine = machine.name
                    candidate.duration = candidate.machine[machine.name]
                    candidate.endTime = candidate.startTime + candidate.duration
                    machine.currentTime = candidate.endTime
                    machine.assignedOpera.append(candidate)
                    # time[machine.currentTime] = {}
                    if machine.currentTime not in list(time.keys()):
                        time[machine.currentTime] = 1
                    else:
                        time[machine.currentTime] += 1
                    jobsListToExport.append(candidate)
        del time[0]
    else:
        idelMachines = []
        t = list(time.keys())[0]
        for machine in machinesList:
            if float(t) >= machine.currentTime:
                idelMachines.append(machine)
        for mac in idelMachines:
            macName = mac.name
            waitingOperations = []
            for task in aJobsList:
                if macName in list(task.machine.keys()) and not task.completed:
                    if task.idOperation == 1:
                        waitingOperations.append(task) 
                    else:
                        operaList = sameJobOpera[task.itinerary]
                        task_id = task.idOperation
                        prev_operation = operaList[task_id-2]
                        if prev_operation.completed and prev_operation.endTime <= float(t):
                            waitingOperations.append(task)
            if len(waitingOperations):
                itinerary = waitingOperations[0].itinerary
                job_id = waitingOperations[0].idOperation
                same_job_list = sameJobOpera[itinerary]
                last_job_id = same_job_list[-1].idOperation
                max_no_process_num = last_job_id - job_id + 1
                incomingOperation = waitingOperations[0]
                for task in waitingOperations[1:]:
                    itinerary = task.itinerary
                    job_id = task.idOperation
                    same_job_list = sameJobOpera[itinerary]
                    last_job_id = same_job_list[-1].idOperation
                    no_process_num = last_job_id - job_id + 1
                    if no_process_num > max_no_process_num:
                        max_no_process_num = no_process_num
                        incomingOperation = task
                if incomingOperation not in jobsListToExport:
                    incomingOperation.startTime = float(t)
                    incomingOperation.completed = True
                    incomingOperation.duration = incomingOperation.machine[mac.name]
                    incomingOperation.assignedMachine = mac.name
                    mac.assignedOpera.append(incomingOperation)
                    mac.currentTime = incomingOperation.getEndTime()
                    # time[mac.currentTime] = {}
                    if mac.currentTime not in list(time.keys()):
                        time[mac.currentTime] = 1
                    else:
                        time[mac.currentTime] += 1
                    jobsListToExport.append(incomingOperation)
        del time[t]
        time = SortedDict(time)
        aJobsList,_ = computingCompletedRatio(aJobsList)
    return jobsListToExport,aJobsList,time

def algorithmSR(aJobsList, machinesList, time):
    '''
    Short Remaining Processing Time
    :param aJobList:
    :param time:
    :return:
    '''
    # print('algorithmSR')
    waitingOperations = {}
    machinesName = []
    jobsListToExport = []
    sameJobList = sameJob_op(aJobsList)
   
    rPRT = {}

    for machine in machinesList:
        machinesName.append(machine.name)
 
    for jobName, operations in sameJobList.items():
        rPRT[jobName] = []
        pRT = 0
        for oper in operations:
            if not oper.completed:
                avaiMac = oper.machine
                sorted(avaiMac,key=lambda j:j[1])
                macName = list(avaiMac.keys())[0]
                pRT += avaiMac[macName]
        rPRT[jobName].append(pRT)

    currentTime = list(time.keys())[0]
    # if not time:
    if currentTime == 0:
        for machine in machinesList:
            waitingOperations = []
            for job in aJobsList:
                if job.idOperation == 1 and machine.name in list(job.machine.keys()):
                    waitingOperations.append(job)
            process = False
            if len(waitingOperations) == 1:
                process = True
                incomingOpera = waitingOperations[0]

            if len(waitingOperations) > 1:
                all_RPRT = []
                process = True
                for oper in waitingOperations:
                    jobitinerary = oper.itinerary
                    rprt = rPRT[jobitinerary]
                    all_RPRT.append(rprt)
                min_index = np.argmin(all_RPRT)
                incomingOpera = waitingOperations[min_index]
            if process and incomingOpera not in jobsListToExport:
                incomingOpera.startTime = 0
                incomingOpera.completed = True
                incomingOpera.duration = incomingOpera.machine[machine.name]
                incomingOpera.endTime = incomingOpera.startTime + incomingOpera.duration
                incomingOpera.assignedMachine = machine.name
                machine.assignedOpera.append(incomingOpera)
                machine.currentTime = incomingOpera.endTime
                # time[machine.currentTime] = []
                if machine.currentTime not in list(time.keys()):
                    time[machine.currentTime] = 1
                else:
                    time[machine.currentTime] += 1
                jobsListToExport.append(incomingOpera)
        del time[0]

    else:
        idelMachines = []
        t = list(time.keys())[0]
        for machine in machinesList:
            if float(t) >= machine.currentTime:
                idelMachines.append(machine)
        for mac in idelMachines:
            macName = mac.name
            waitingOperations = []
            for task in aJobsList:
                if macName in list(task.machine.keys()) and not task.completed:
                    task_id = task.idOperation
                    if task_id == 1:
                        waitingOperations.append(task)
                    else:
                        task_itinerary = task.itinerary
                        same_itinerary_operations = sameJobList[task_itinerary]
                        pre_tassk = same_itinerary_operations[task_id - 2]
                        if pre_tassk.completed and pre_tassk.endTime <= float(t):
                            waitingOperations.append(task) 
            if len(waitingOperations) == 1 and waitingOperations[0] not in jobsListToExport:
                waitingOperations[0].startTime = float(t)
                waitingOperations[0].completed = True
                waitingOperations[0].duration = waitingOperations[0].machine[macName]
                waitingOperations[0].endTime = waitingOperations[0].startTime + waitingOperations[0].duration
                waitingOperations[0].assignedMachine = macName
                mac.assignedOpera.append(waitingOperations[0])
                mac.currentTime = waitingOperations[0].endTime
                # time[mac.currentTime] = []
                if mac.currentTime not in list(time.keys()):
                    time[mac.currentTime] = 1
                else:
                    time[mac.currentTime] += 1
                jobsListToExport.append(waitingOperations[0])
            if len(waitingOperations) > 1:
                all_RPRT = []
                for oper in waitingOperations:
                    jobitinerary = oper.itinerary
                    rprt = rPRT[jobitinerary]
                    all_RPRT.append(rprt)
                min_index = np.argmin(all_RPRT)
                incomingOpera = waitingOperations[min_index]
                if incomingOpera not in jobsListToExport:
                    incomingOpera.startTime = float(t)
                    incomingOpera.completed = True
                    incomingOpera.duration = incomingOpera.machine[macName]
                    incomingOpera.endTime = incomingOpera.startTime + incomingOpera.duration
                    incomingOpera.assignedMachine = macName
                    mac.assignedOpera.append(incomingOpera)
                    mac.currentTime = incomingOpera.endTime
                    # time[mac.currentTime] = []
                    if mac.currentTime not in list(time.keys()):
                        time[mac.currentTime] = 1
                    else:
                        time[mac.currentTime] += 1
                    jobsListToExport.append(incomingOpera)
        del time[t]
        time = SortedDict(time)
        aJobsList, _ = computingCompletedRatio(aJobsList)
    return jobsListToExport, aJobsList, time

def algorithmLR(aJobsList, machinesList, time):
    '''
    Longest Remaining Processing Time
    :param aJobsList:
    :param time:
    :return:
    '''
    # print('algorithmLR')
    waitingOperations = {}
    machinesName = []
    jobsListToExport = []
    sameJobList = sameJob_op(aJobsList)

    rPRT = {}

    for machine in machinesList:
        machinesName.append(machine.name)

    for jobName, operations in sameJobList.items():
        rPRT[jobName] = []
        pRT = 0
        for oper in operations:
            if not oper.completed:
                avaiMac = oper.machine
                sorted(avaiMac,key=lambda j:j[1],reverse=True)
                macName = list(avaiMac.keys())[0]
                pRT += avaiMac[macName]
        rPRT[jobName].append(pRT)

    currentTime = list(time.keys())[0]
    # if not time:
    if currentTime == 0:
        for machine in machinesList:
            waitingOperations = []
            for job in aJobsList:
                if not job.completed and machine.name in list(job.machine.keys()) and job.idOperation == 1:
                    waitingOperations.append(job)
            process = False
            if len(waitingOperations) == 1:
                process = True
                incomingOpera = waitingOperations[0]

            if len(waitingOperations) > 1:
                all_RPRT = []
                process = True
                for oper in waitingOperations:
                    jobitinerary = oper.itinerary
                    rprt = rPRT[jobitinerary]
                    all_RPRT.append(rprt)
                max_index = np.argmax(all_RPRT)
                incomingOpera = waitingOperations[max_index]
            if process and incomingOpera not in jobsListToExport:
                incomingOpera.startTime = currentTime
                incomingOpera.completed = True
                incomingOpera.duration = incomingOpera.machine[machine.name]
                incomingOpera.endTime = incomingOpera.startTime + incomingOpera.duration
                incomingOpera.assignedMachine = machine.name
                machine.assignedOpera.append(incomingOpera)
                machine.currentTime = incomingOpera.endTime
                # time[machine.currentTime] = []
                if machine.currentTime not in list(time.keys()):
                    time[machine.currentTime] = 1
                else:
                    time[machine.currentTime] += 1
                jobsListToExport.append(incomingOpera)
        del time[0]
    else:
        idelMachines = []
        t = list(time.keys())[0]
        for machine in machinesList:
            if float(t) >= machine.currentTime:
                idelMachines.append(machine)
        for mac in idelMachines:
            macName = mac.name
            waitingOperations = []
            for task in aJobsList:
                if macName in list(task.machine.keys()) and not task.completed:
                    job_id = task.idOperation
                    if job_id == 1:
                        waitingOperations.append(task)  
                    else:
                        task_itinerary = task.itinerary
                        same_itinerary_operations = sameJobList[task_itinerary]
                        pre_task = same_itinerary_operations[job_id-2]
                        if pre_task.completed and pre_task.endTime <= float(t):
                            waitingOperations.append(task)

            if len(waitingOperations) == 1 and waitingOperations[0] not in jobsListToExport:
                waitingOperations[0].startTime = float(t)
                waitingOperations[0].completed = True
                waitingOperations[0].duration = waitingOperations[0].machine[macName]
                waitingOperations[0].endTime = waitingOperations[0].startTime + waitingOperations[0].duration
                waitingOperations[0].assignedMachine = macName
                mac.assignedOpera.append(waitingOperations[0])
                mac.currentTime = waitingOperations[0].endTime
                # time[mac.currentTime] = []
                if mac.currentTime not in list(time.keys()):
                    time[mac.currentTime] = 1
                else:
                    time[mac.currentTime] += 1
                jobsListToExport.append(waitingOperations[0])
            if len(waitingOperations) > 1:
                all_RPRT = []
                for oper in waitingOperations:
                    jobitinerary = oper.itinerary
                    rprt = rPRT[jobitinerary]
                    all_RPRT.append(rprt)
                max_index = np.argmin(all_RPRT)
                incomingOpera = waitingOperations[max_index]
                if incomingOpera not in jobsListToExport:
                    incomingOpera.startTime = float(t)
                    incomingOpera.completed = True
                    incomingOpera.duration = incomingOpera.machine[macName]
                    incomingOpera.endTime = incomingOpera.startTime + incomingOpera.duration
                    incomingOpera.assignedMachine = macName
                    mac.assignedOpera.append(incomingOpera)
                    mac.currentTime = incomingOpera.endTime
                    # time[mac.currentTime] = []
                    if mac.currentTime not in list(time.keys()):
                        time[mac.currentTime] = 1
                    else:
                        time[mac.currentTime] += 1
                    jobsListToExport.append(incomingOpera)
        del time[t]
        time = SortedDict(time)
        aJobsList, _ = computingCompletedRatio(aJobsList)
    return jobsListToExport, aJobsList, time



def randomSolution(aJobsList, machinesList, time):
    '''
    Longest Remaining Processing Time
    :param aJobsList:
    :param time:
    :return:
    '''
    # print('algorithmLR')
    waitingOperations = {}
    machinesName = []
    jobsListToExport = []
    sameJobList = sameJob_op(aJobsList)
    rPRT = {}

    for machine in machinesList:
        machinesName.append(machine.name)

    for jobName, operations in sameJobList.items():
        rPRT[jobName] = []
        pRT = 0
        for oper in operations:
            if not oper.completed:
                avaiMac = oper.machine
                sorted(avaiMac,key=lambda j:j[1],reverse=True)
                macName = list(avaiMac.keys())[0]
                pRT += avaiMac[macName]
        rPRT[jobName].append(pRT)

    currentTime = list(time.keys())[0]
    # if not time:
    if currentTime == 0:
        for machine in machinesList:
            waitingOperations = []
            for job in aJobsList:
                if not job.completed and machine.name in list(job.machine.keys()) and job.idOperation == 1:
                    waitingOperations.append(job)
            process = False
            if len(waitingOperations) == 1:
                process = True
                incomingOpera = waitingOperations[0]

            if len(waitingOperations) > 1:
                all_RPRT = []
                process = True
                random_index = random.randint(0, len(waitingOperations) - 1)
                incomingOpera = waitingOperations[random_index]
            if process and incomingOpera not in jobsListToExport:
                incomingOpera.startTime = currentTime
                incomingOpera.completed = True
                incomingOpera.duration = incomingOpera.machine[machine.name]
                incomingOpera.endTime = incomingOpera.startTime + incomingOpera.duration
                incomingOpera.assignedMachine = machine.name
                machine.assignedOpera.append(incomingOpera)
                machine.currentTime = incomingOpera.endTime
                # time[machine.currentTime] = []
                if machine.currentTime not in list(time.keys()):
                    time[machine.currentTime] = 1
                else:
                    time[machine.currentTime] += 1
                jobsListToExport.append(incomingOpera)
        del time[0]
    else:
        idelMachines = []
        t = list(time.keys())[0]
        for machine in machinesList:
            if float(t) >= machine.currentTime:
                idelMachines.append(machine)
        for mac in idelMachines:
            macName = mac.name
            waitingOperations = []
            for task in aJobsList:
                if macName in list(task.machine.keys()) and not task.completed:
                    job_id = task.idOperation
                    if job_id == 1:
                        waitingOperations.append(task)  
                    else:
                        task_itinerary = task.itinerary
                        same_itinerary_operations = sameJobList[task_itinerary]
                        pre_task = same_itinerary_operations[job_id-2]
                        if pre_task.completed and pre_task.endTime <= float(t):
                            waitingOperations.append(task)

            if len(waitingOperations) == 1 and waitingOperations[0] not in jobsListToExport:
                waitingOperations[0].startTime = float(t)
                waitingOperations[0].completed = True
                waitingOperations[0].duration = waitingOperations[0].machine[macName]
                waitingOperations[0].endTime = waitingOperations[0].startTime + waitingOperations[0].duration
                waitingOperations[0].assignedMachine = macName
                mac.assignedOpera.append(waitingOperations[0])
                mac.currentTime = waitingOperations[0].endTime
                # time[mac.currentTime] = []
                if mac.currentTime not in list(time.keys()):
                    time[mac.currentTime] = 1
                else:
                    time[mac.currentTime] += 1
                jobsListToExport.append(waitingOperations[0])
            if len(waitingOperations) > 1:
                random_index = random.randint(0, len(waitingOperations)-1)
                incomingOpera = waitingOperations[random_index]
                if incomingOpera not in jobsListToExport:
                    incomingOpera.startTime = float(t)
                    incomingOpera.completed = True
                    incomingOpera.duration = incomingOpera.machine[macName]
                    incomingOpera.endTime = incomingOpera.startTime + incomingOpera.duration
                    incomingOpera.assignedMachine = macName
                    mac.assignedOpera.append(incomingOpera)
                    mac.currentTime = incomingOpera.endTime
                    # time[mac.currentTime] = []
                    if mac.currentTime not in list(time.keys()):
                        time[mac.currentTime] = 1
                    else:
                        time[mac.currentTime] += 1
                    jobsListToExport.append(incomingOpera)
        del time[t]
        time = SortedDict(time)
        aJobsList, _ = computingCompletedRatio(aJobsList)
    return jobsListToExport, aJobsList, time

def computingCompletedRatio(aJobsList):
    completedRatio = {}

    operasOfJob = sameJob_op(aJobsList)

    for jobName, tasks in operasOfJob.items():
        completedNum = 0
        for opera in tasks:
            if opera.completed:
                completedNum += 1
        completedRatio[jobName] = completedNum / len(tasks)
        for opera in tasks:
            opera.completedRatio = completedRatio[jobName]
    return aJobsList,operasOfJob

