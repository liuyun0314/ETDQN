import random
import numpy as np
from clJob import Job
from clMachine import Machine

def insert_instance_generator(already_job_num, total_machine_num, insert_job_num, max_process_time, max_operation_num):
    '''
    randomly generate insert instance
    :param already_job_num:
    :param total_machine_num:
    :param insert_job_num:
    :return:
    '''
    insert_job = []
    itineraryColors = []
    for i in range(insert_job_num):
        operation_num = np.random.randint(1, max_operation_num)
        idItinerary = already_job_num + i + 1
        itinerary = 'itinerary' + str(idItinerary)
        operationsList = []
        for j in range(operation_num):
            idOperation = j + 1
            candidated_machine_num = np.random.randint(1, 6)
            sample_index = np.random.choice(range(total_machine_num), candidated_machine_num, replace=False)
            # processing_time = np.random.randint(1, 100, 5)
            processing_time = np.random.randint(1, max_process_time, 5)
            pastelFactor = random.uniform(0, 1)
            itineraryColors.append(
                generate_new_color(itineraryColors, pastelFactor))
            machine = {}
            count = 0
            for m in sample_index:
                machine_name = 'M' + str(m + 1)
                machine[machine_name] = processing_time[count]
                count += 1
            operation = Job(itinerary, itineraryColors[0], idOperation, idItinerary, machine, 0)
            operationsList.append(operation)
        insert_job.append(operationsList)
    return insert_job

def color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])
def get_random_color(pastel_factor=0.5):
    return [(x + pastel_factor) / (1.0 + pastel_factor) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]]
def generate_new_color(existing_colors, pastel_factor=0.5):
    """Generate new color if not exist in existing array of colors"""
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color