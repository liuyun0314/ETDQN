from parseData import parseData
from algorithms import prepareJobs
import pylab
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict
from sortedcollections import SortedDict
from get_fjs import readData

# Edge types
CONJUNCTIVE_TYPE = 0
DISJUNCTIVE_TYPE = 1

# Edge directions
FORWARD = 0   
BIDIRECTION = 1   

def creatDisjunctiveGraph(taskList, machinesList) -> nx.OrderedDiGraph:

    g = nx.MultiDiGraph()
    processedOperation = []  
    waitingOperations = []  
    DisjunctiveNode = []
    for opera in taskList:
        if opera.completed:
            processedOperation.append(opera)
        else:
            waitingOperations.append(opera)

    for task in taskList:
        node_attri = OrderedDict()
        # node attri
        node_attri['itinerary'] = task.itinerary
        node_attri['idItinerary'] = task.idItinerary
        node_attri['idOperation'] = task.idOperation
        node_attri['startTime'] = task.startTime
        node_attri['duration'] = task.duration
        node_attri['endTime'] = task.endTime
        node_attri['completed'] = task.completed
        node_attri['priority'] = task.priority
        node_attri['assignedMachine'] = task.assignedMachine
        taskOfJob = task.idItinerary  
        taskID = task.idOperation  
        taskName = 'O' + str(taskOfJob) + str(taskID)
        g.add_node(node_for_adding=taskName, keyword=task)

 
    disjunctive_tasks = disjunctive_ops(waitingOperations, machinesList)


    sameJobOpts = sameJob_op(taskList)
    for jobName, tasks in sameJobOpts.items():
        tasks.sort(key=lambda j: j.idOperation)

    # conjunctive arcs
    conjunctive_arcs = []
    for job, jobTask in sameJobOpts.items():
        for i in range(len(jobTask) - 1):
            taskOfJob = jobTask[i].idItinerary 
            taskID = jobTask[i].idOperation 
            jobTaskName = 'O' + str(taskOfJob) + str(taskID)

            nextTaskOfJob = jobTask[i+1].idItinerary 
            nextTaskID = jobTask[i+1].idOperation 
            jobNextTaskName = 'O' + str(nextTaskOfJob) + str(nextTaskID)

            edge = (jobTaskName, jobNextTaskName, 0)
            conjunctive_arcs.append(edge)
    g.add_weighted_edges_from(conjunctive_arcs, direction=FORWARD, type=CONJUNCTIVE_TYPE)

    for machine in machinesList:
        direction_arcs = []
        assiOperas = machine.assignedOpera
        # for assiOpera in machine.assignedOpera:
        if len(assiOperas) > 1:
            for i in range(len(assiOperas) - 1):
                preAssiJob = assiOperas[i].idItinerary
                preOperaId = assiOperas[i].idOperation
                preAssiOpera = 'O' + str(preAssiJob) + str(preOperaId)

                assiJob = assiOperas[i+1].idItinerary
                operaId = assiOperas[i+1].idOperation
                AssiOpera = 'O' + str(assiJob) + str(operaId)

                edge = (preAssiOpera, AssiOpera, assiOperas[i].duration)
                direction_arcs.append(edge)
            # g.add_weighted_edges_from(direction_arcs)

        if len(assiOperas):
            assiJob = assiOperas[-1].idItinerary
            operaId = assiOperas[-1].idOperation
            AssiOpera = 'O' + str(assiJob) + str(operaId)
            for operation in disjunctive_tasks[machine.name]:
                operaJob = operation.idItinerary
                operaId = operation.idOperation
                operaName = 'O' + str(operaJob) + str(operaId)

                edge = (AssiOpera, operaName, assiOperas[-1].duration)
                direction_arcs.append(edge)
        g.add_weighted_edges_from(direction_arcs, direction=FORWARD, type=DISJUNCTIVE_TYPE, machine=machine)


    for machine, disj_task in disjunctive_tasks.items():
        disjunctive_edges = []
        for i in range(len(disj_task) - 1):
            for j in range(i+1, len(disj_task)):
                taskOfJob = disj_task[i].idItinerary  
                taskID = disj_task[i].idOperation  
                disjTaskName = 'O' + str(taskOfJob) + str(taskID)

                nextTaskOfJob = disj_task[j].idItinerary
                nextTaskID = disj_task[j].idOperation
                disjNextTaskName = 'O' + str(nextTaskOfJob) + str(nextTaskID)


                edge = (disjTaskName, disjNextTaskName,disj_task[i].duration)
                disjunctive_edges.append(edge)
        g.add_weighted_edges_from(disjunctive_edges,direction=BIDIRECTION,type=DISJUNCTIVE_TYPE, machine=machine)

    return g

def disjunctive_ops(TaskList, machineList):
    disjunctive_ops = {}    
    for machine in machineList:
        disjunctive_ops[machine.name] = []
        for task in TaskList:
            if machine.name in task.machine:
                disjunctive_ops[machine.name].append(task)
    return disjunctive_ops

def sameJob_op(TaskList):

    sameJobOpts = {}
    for task in TaskList:
        if task.itinerary not in sameJobOpts:
            sameJobOpts[task.itinerary] = []
        sameJobOpts[task.itinerary].append(task)
    for jobName, operations in sameJobOpts.items():
        # sorted(operations,key=lambda j:j.idOperation)
        operations.sort(key=lambda j:j.idOperation)
    return sameJobOpts

def plot_graph(g, taskList, machineList, draw: bool = True,
               node_type_color_dict: dict = None,
               edge_type_color_dict: dict = None,
               half_width=None,
               half_height=None,
               **kwargs):

    node_colors, _ = get_node_color_map(g, taskList, machineList, node_type_color_dict)
    edge_colors = get_edge_color_map(g, machineList, edge_type_color_dict)
    pos = calc_positions(g, taskList,machineList, half_width, half_height)

    if not kwargs:
        kwargs['figsize'] = (10, 5)
        kwargs['dpi'] = 300

    fig = plt.figure(**kwargs)
    ax = fig.add_subplot(1, 1, 1)

    machineNames = []
    for mac in machineList:
        machineNames.append(mac.name)

    nx.draw(g, pos,
            node_color=node_colors,
            edge_color=edge_colors,
            with_labels=True,
            # style='dashed',  
            font_size = 4,
            node_size = 50,
            width=0.5,
            arrowsize=5,
            label=False,
            ax=ax)
    if draw:
        pylab.title('The disjunctive graph of a instance',fontsize = 9)
        # plt.legend(color_kind,loc='best',fontsize=5)
        plt.show()
    else:
        return fig, ax

def get_node_color_map(g, taskList, machineList, node_type_color_dict=None):
    if node_type_color_dict is None:
        node_type_color_dict = OrderedDict()

    disjunctive_tasks = disjunctive_ops(taskList, machineList)
    colors = {}
    colorSets = []
    node_colors = ['#e9e7ef', '#f9906f', '#44cef6', '#2add9c', '#fff143', '#cca4e3', '#725e80', '#3de1ad', '#ef7a82',
                   '#ae7000', '#057748', '#f47983', '#ca6924', '#b35c44','#c0ebd7','#db5a6b','#3eede7','#f9908f','#eacd76','#bce672']
    i = 0
    for machine, tasks in disjunctive_tasks.items():
        i += 1
        colors[machine] = node_colors[i]
        # colors[machine] = randonColor()

    for n in g.nodes:
        node = g.nodes[n]['keyword']
        if not node.completed:
            node_type_color_dict['uncompleted'] = node_colors[0]
            colorSets.append(node_type_color_dict['uncompleted'])
            continue
        mac = node.assignedMachine
        node_type_color_dict[mac] = colors[mac]
        colorSets.append(node_type_color_dict[mac])
    return colorSets, colors

def get_edge_color_map(g, machinesList, edge_type_color_dict=None):

    if edge_type_color_dict is None:
        edge_type_color_dict = OrderedDict()
        edge_type_color_dict[CONJUNCTIVE_TYPE] = '#DCDCDC'
        edge_colors =['#ff4777','#f9906f','#44cef6','#2add9c','#fff143','#cca4e3','#725e80','#3de1ad','#ef7a82', '#ae7000', '#057748','#f47983','#ca6924','#b35c44','#e9e7ef','#c0ebd7','#db5a6b','#3eede7','#f9908f','#eacd76','#bce672']
        i = 0
        for machine in machinesList:
            # edge_type_color_dict[machine] = randonColor()
            edge_type_color_dict[machine.name] = edge_colors[i]
            i += 1

    colors = []
    for e in g.edges:
        if g.edges[e]['type'] == CONJUNCTIVE_TYPE:
            colors.append(edge_type_color_dict[CONJUNCTIVE_TYPE])
        else:
            for machine in machinesList:
                if str(g.edges[e]['machine']) == machine.name:
                    colors.append(edge_type_color_dict[machine.name])

    return colors

def calc_positions(g, taskList, machineList, half_width=None, half_height=None):

    nodeNum = len(g.nodes)
    pos_dict = OrderedDict()
    if half_width is None:
        half_width = 30   # 30
    if half_height is None:
        half_height = 10     # 10

    sameJobOpts = sameJob_op(taskList)
    sameJobOpts = SortedDict(sameJobOpts)
    taskNumForJob = []
    i = 0
    for jobName, tasks in sameJobOpts.items():
        taskOfJob = tasks[0].idItinerary  
        tasks.sort(key=lambda j: j.idOperation)
        taskNum = len(tasks)
        taskNumForJob.append(taskNum)
        for j in range(taskNum):
            taskID = tasks[j].idOperation  
            TaskName = 'O' + str(taskOfJob) + str(taskID)
            pos_dict[TaskName] = np.array((j, i))
        i += 1
    return pos_dict

def randonColor():
    color_list = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ''
    for i in range(6):
        color_number = color_list[random.randint(0,15)]
        color += color_number
    color = '#' + color
    return color
