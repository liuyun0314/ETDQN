
import numpy as np

def newJobsArrival(jobNums):
    '''
    :param jobNums: the number of new jobs
    :return: new jobs arrival times
    '''

    mu = 88
    PM = 0.5
    P = 0.95
    begin_time = 0
    arrival_times = []
    # inter_arrival = interArrival(mu, PM, P, jobNums)
    # print(inter_arrival)
    # arrival_times = np.cumsum(inter_arrival) 
    for i in range(jobNums):
        inter_arrival = interArrival(mu, PM, P)
        while inter_arrival == 0:
            inter_arrival = interArrival(mu, PM, P)
        arrival_time = begin_time + inter_arrival
        arrival_times.append(arrival_time)
        begin_time = arrival_times[-1]
    return arrival_times

def interArrival(mu, PM, P):
    '''
    :param mu: the average processing time of the machines
    :param P: probability of a job visiting a machine
    :jobNums: the number of new jobs
    :return: interval between new jobs
    '''
    lamada = mu * PM / P
    inter_arrival = np.random.poisson(lamada)
    return inter_arrival

print(newJobsArrival(45))